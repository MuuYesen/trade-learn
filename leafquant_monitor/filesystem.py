"""Narrow private-file and proof-retirement primitives, with no framework dependency.

Windows uses real handle security, FlushFileBuffers and same-directory
MoveFileExW(WRITE_THROUGH), never a pretend directory fsync or copy fallback.
https://learn.microsoft.com/windows/win32/api/aclapi/nf-aclapi-getsecurityinfo
https://learn.microsoft.com/windows/win32/api/securitybaseapi/nf-securitybaseapi-getace
https://learn.microsoft.com/windows/win32/api/winbase/nf-winbase-movefileexw
"""
import os
from pathlib import Path
import stat
from functools import lru_cache


@lru_cache(maxsize=1)
def _api():
    import ctypes as c
    from ctypes import wintypes as w
    kernel=c.WinDLL('kernel32',use_last_error=True)
    security=c.WinDLL('advapi32',use_last_error=True)
    pointer=c.c_void_p
    signatures=[(kernel,'GetCurrentProcess',[],w.HANDLE),(kernel,'GetCurrentThread',[],w.HANDLE),
        (kernel,'CloseHandle',[w.HANDLE],w.BOOL),(kernel,'LocalFree',[pointer],pointer),
        (kernel,'MoveFileExW',[w.LPCWSTR,w.LPCWSTR,w.DWORD],w.BOOL),
        (kernel,'CreateFileW',[w.LPCWSTR,w.DWORD,w.DWORD,pointer,w.DWORD,w.DWORD,w.HANDLE],w.HANDLE),
        (kernel,'FlushFileBuffers',[w.HANDLE],w.BOOL),
        (kernel,'GetFileInformationByHandleEx',[w.HANDLE,c.c_int,pointer,w.DWORD],w.BOOL),
        (security,'OpenProcessToken',[w.HANDLE,w.DWORD,c.POINTER(w.HANDLE)],w.BOOL),
        (security,'OpenThreadToken',[w.HANDLE,w.DWORD,w.BOOL,c.POINTER(w.HANDLE)],w.BOOL),
        (security,'GetTokenInformation',[w.HANDLE,c.c_int,pointer,w.DWORD,c.POINTER(w.DWORD)],w.BOOL),
        (security,'GetSecurityInfo',[w.HANDLE,c.c_int,w.DWORD,c.POINTER(pointer),pointer,c.POINTER(pointer),pointer,c.POINTER(pointer)],w.DWORD),
        (security,'GetAce',[pointer,w.DWORD,c.POINTER(pointer)],w.BOOL),
        (security,'ConvertSidToStringSidW',[pointer,c.POINTER(pointer)],w.BOOL)]
    for library,name,args,result in signatures:
        function=getattr(library,name);function.argtypes=args;function.restype=result
    return c,w,kernel,security


def _sid_string(sid):
    c,_,kernel,security=_api();output=c.c_void_p()
    if not security.ConvertSidToStringSidW(sid,c.byref(output)):raise c.WinError(c.get_last_error())
    try:return c.wstring_at(output)
    finally:kernel.LocalFree(output)


def _token_sids():
    c,w,kernel,security=_api();token=w.HANDLE()
    if not security.OpenThreadToken(kernel.GetCurrentThread(),8,True,c.byref(token)):
        if c.get_last_error()!=1008:raise c.WinError(c.get_last_error())
        if not security.OpenProcessToken(kernel.GetCurrentProcess(),8,c.byref(token)):raise c.WinError(c.get_last_error())
    try:
        result=[]
        for kind in (1,4):  # TOKEN_USER and the current token's default TOKEN_OWNER.
            size=w.DWORD();security.GetTokenInformation(token,kind,None,0,c.byref(size))
            if not 0<size.value<=65536:raise ValueError('bounded Windows token required')
            buffer=c.create_string_buffer(size.value)
            if not security.GetTokenInformation(token,kind,buffer,size,c.byref(size)):raise c.WinError(c.get_last_error())
            result.append(_sid_string(c.c_void_p.from_buffer(buffer)))
        return result
    finally:kernel.CloseHandle(token)


def _check_windows_handle(handle):
    c,_,kernel,security=_api();owner=c.c_void_p();acl=c.c_void_p();descriptor=c.c_void_p()
    code=security.GetSecurityInfo(handle,1,1|4,c.byref(owner),None,c.byref(acl),None,c.byref(descriptor))
    if code:raise c.WinError(code)
    try:
        user,default_owner=_token_sids()
        trusted={user,'S-1-5-18','S-1-5-32-544'}
        if not owner or _sid_string(owner) not in ({user,default_owner}&trusted) or not acl:
            raise ValueError('private Windows owner and non-null DACL required')
        # ACL header is BYTE revision, BYTE pad, WORD size, WORD count, WORD pad.
        count=c.c_ushort.from_address(acl.value+4).value
        for index in range(count):
            ace=c.c_void_p()
            if not security.GetAce(acl,index,c.byref(ace)):raise c.WinError(c.get_last_error())
            kind=c.c_ubyte.from_address(ace.value).value
            flags=c.c_ubyte.from_address(ace.value+1).value
            size=c.c_ushort.from_address(ace.value+2).value
            if kind==1:continue  # Deny ACEs cannot broaden access.
            if kind!=0 or size<12:raise ValueError('unsupported Windows access-control grant')
            mask=c.c_uint32.from_address(ace.value+4).value
            sid=_sid_string(c.c_void_p(ace.value+8))
            # Inherit-only CREATOR OWNER becomes the actual owner on a child.
            if mask and sid not in trusted and not (sid=='S-1-3-0' and flags&8):
                raise ValueError('Windows file grants access outside its private owner')
    finally:kernel.LocalFree(descriptor)


def verify_private_fd(fd):
    info=os.fstat(fd)
    if not stat.S_ISREG(info.st_mode) or info.st_nlink!=1 or getattr(info,'st_file_attributes',0)&0x400:
        raise ValueError('private regular non-reparse file required')
    if os.name=='nt':
        import msvcrt
        _check_windows_handle(msvcrt.get_osfhandle(fd))
    elif stat.S_IMODE(info.st_mode)!=0o600 or info.st_uid!=os.getuid():
        raise ValueError('private owner-only file required')
    return file_stat(fd)


def check_private_path(path):
    before=path_stat(path)
    if not stat.S_ISREG(before.st_mode) or getattr(before,'st_file_attributes',0)&0x400:
        raise ValueError('private non-reparse file required')
    fd=open_read_fd(path)
    try:
        info=verify_private_fd(fd)
        if (before.st_dev,before.st_ino)!=(info.st_dev,info.st_ino):raise ValueError('private file replaced during validation')
    finally:os.close(fd)


def preflight_sender(settings, *, credential_files=True):
    """Reject unsupported/insecure Windows sending before constructing a recorder."""
    if os.name!='nt':return
    import json
    if credential_files:
        check_private_path(settings.sender_config)
        value=json.loads(Path(settings.sender_config).read_text())
        credential=Path(value['credentials_file'])
        if not credential.is_absolute():raise ValueError('absolute private sender credential path required')
        check_private_path(credential)
    # Inspect the nearest existing directory through a real backup-semantics
    # handle. Future children inherit its DACL; every opened file is rechecked.
    directory=Path(settings.state_dir).absolute()
    while not directory.exists():directory=directory.parent
    check_private_directory(directory)


def check_private_directory(directory):
    if os.name!='nt':return
    directory=Path(directory).absolute()
    c,w,kernel,_=_api()
    create=kernel.CreateFileW;create.argtypes=[w.LPCWSTR,w.DWORD,w.DWORD,c.c_void_p,w.DWORD,w.DWORD,w.HANDLE];create.restype=w.HANDLE
    handle=create(str(directory),0x20000,7,None,3,0x02000000|0x00200000,None)
    if handle==c.c_void_p(-1).value:raise c.WinError(c.get_last_error())
    try:
        if getattr(directory.lstat(),'st_file_attributes',0)&0x400:raise ValueError('private non-reparse state directory required')
        _check_windows_handle(handle)
    finally:kernel.CloseHandle(handle)


def move_write_through(source,destination):
    source,destination=Path(source).absolute(),Path(destination).absolute()
    if source.parent!=destination.parent:raise ValueError('retirement must stay in one directory')
    c,_,kernel,_=_api()
    # No REPLACE_EXISTING and no COPY_ALLOWED: collisions retain both originals.
    if not kernel.MoveFileExW(str(source),str(destination),8):raise c.WinError(c.get_last_error())


def sync_committed_database(path):
    """SQLite FULL commits first; Windows then flushes its actual file handle."""
    if os.name!='nt':
        fd=os.open(Path(path).parent,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)
        return
    fd=os.open(path,os.O_RDWR)
    try:
        verify_private_fd(fd)
        import msvcrt
        c,_,kernel,_=_api()
        if not kernel.FlushFileBuffers(msvcrt.get_osfhandle(fd)):raise c.WinError(c.get_last_error())
    finally:os.close(fd)


def database_identity(value):
    """SQLite INTEGER is signed; prefixed TEXT preserves larger unsigned IDs."""
    if type(value) is not int or not 0<=value<2**128:raise ValueError('bounded non-negative file identity required')
    return value if value<2**63 else 'u:'+str(value)


def original_identity(value):
    if type(value) is int and 0<=value<2**63:return value
    if isinstance(value,str) and value.startswith('u:'):
        digits=value[2:]
        if digits.isascii() and digits.isdecimal() and not digits.startswith('0'):
            number=int(digits)
            if 2**63<=number<2**128:return number
    raise ValueError('invalid persisted file identity')


class _CanonicalStat:
    def __init__(self, original, device, inode):
        self._original=original;self.st_dev=device;self.st_ino=inode
    def __getattr__(self,name):return getattr(self._original,name)


def file_stat(fd):
    """Version-independent Windows FILE_ID_INFO; Unix retains its native stat."""
    info=os.fstat(fd)
    if os.name!='nt':return info
    import msvcrt
    c,_,kernel,_=_api()
    class FileId(c.Structure):
        _fields_=[('volume',c.c_uint64),('identifier',c.c_ubyte*16)]
    identity=FileId()
    if not kernel.GetFileInformationByHandleEx(msvcrt.get_osfhandle(fd),18,c.byref(identity),c.sizeof(identity)):
        raise c.WinError(c.get_last_error())
    return _CanonicalStat(info,identity.volume,int.from_bytes(bytes(identity.identifier),'little'))


def path_stat(path):
    before=os.lstat(path)
    if os.name!='nt':return before
    if not stat.S_ISREG(before.st_mode) or getattr(before,'st_file_attributes',0)&0x400:
        raise ValueError('regular non-reparse path required')
    fd=open_read_fd(path)
    try:
        opened=os.fstat(fd)
        # These two OS stat values come from the SAME interpreter. Never mix
        # their version-specific Windows identity with the canonical wire ID.
        if (before.st_dev,before.st_ino)!=(opened.st_dev,opened.st_ino):
            raise ValueError('source path changed while opening')
        return file_stat(fd)
    finally:os.close(fd)



def open_read_fd(path):
    """Pinned exact-byte reader that permits concurrent atomic publication."""
    if os.name!='nt':return os.open(path,os.O_RDONLY|getattr(os,'O_NOFOLLOW',0))
    import msvcrt
    c,_,kernel,_=_api()
    handle=kernel.CreateFileW(str(Path(path).absolute()),0x80000000,7,None,3,0x00200000,None)
    if handle==c.c_void_p(-1).value:raise c.WinError(c.get_last_error())
    try:return msvcrt.open_osfhandle(handle,os.O_RDONLY|os.O_BINARY)
    except BaseException:kernel.CloseHandle(handle);raise
