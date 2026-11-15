"""Native diagnostic logging defaults; independent of durable monitoring records.

Only constructs official NA configuration. It neither touches the filesystem nor
installs process-global Python handlers. NA opens files when its node starts.
"""
from collections.abc import Mapping


_LOGGER_FIELDS = ('stdout_level', 'fileout_level', 'component_levels', 'is_colored',
    'print_config', 'bypass_logging', 'log_components_only', 'file_config',
    'clear_log_file', 'fileout_sync_on_flush', 'buffered_stdout')


def diagnostic_config(options=None, *, native=None, directory='logs'):
    """Resolve Store ``logging`` into an official LoggerConfig, without I/O.

    None supplies TL defaults unless the node has a non-default logger. False
    disables TL's default file only. A LoggerConfig is used unchanged. A mapping
    configures the common console/file/rotation defaults for every adapter.
    """
    from nautilus_trader.common import LoggerConfig, FileWriterConfig, LogLevel
    if isinstance(options, LoggerConfig):
        return options
    if options is False:
        return LoggerConfig(stdout_level=LogLevel.INFO)
    if options is None and native is not None:
        default = LoggerConfig()
        if any(getattr(native, key) != getattr(default, key) for key in _LOGGER_FIELDS):
            return native
    if options is not None and not isinstance(options, Mapping):
        raise TypeError('logging requires None, False, a mapping or official LoggerConfig')
    values = dict(options or {})
    allowed = {'stdout_level', 'fileout_level', 'directory', 'file_name',
               'max_bytes', 'backup_count', 'is_colored'}
    if values.keys() - allowed:
        raise TypeError('unknown diagnostic logging options: ' + ', '.join(sorted(values.keys() - allowed)))
    size = values.get('max_bytes', 10 * 1024 * 1024)
    count = values.get('backup_count', 5)
    if type(size) is not int or size <= 0 or type(count) is not int or count < 0:
        raise ValueError('positive integer max_bytes and nonnegative integer backup_count required')
    folder = values.get('directory', directory)
    name = values.get('file_name', 'strategy.log')
    if not isinstance(folder, str) or not folder.strip() or not isinstance(name, str) or not name.strip():
        raise ValueError('nonempty diagnostic directory and file_name required')
    colored = values.get('is_colored', False)
    if type(colored) is not bool:
        raise TypeError('is_colored must be boolean')
    def level(key):
        value = values.get(key, 'INFO')
        if isinstance(value, str):
            value = getattr(LogLevel, value.upper(), None)
        if not isinstance(value, LogLevel):
            raise ValueError(f'{key} requires an official LogLevel or level name')
        return value
    return LoggerConfig(stdout_level=level('stdout_level'), fileout_level=level('fileout_level'),
        is_colored=colored, file_config=FileWriterConfig(directory=folder, file_name=name,
                                                       file_rotate=(size, count)))
