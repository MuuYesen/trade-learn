from jinja2 import Environment, PackageLoader


def generate_stylesheet(scheme, template='basic.css.j2'):
    '''
    Generates stylesheet with values from scheme
    '''
    import os
    cur_dir_path = os.path.abspath(os.path.dirname(__file__))
    from jinja2 import Environment, FileSystemLoader
    env = Environment(loader=FileSystemLoader(os.path.join(cur_dir_path, '..', 'templates')))
    templ = env.get_template(template)

    css = templ.render(scheme.__dict__)
    return css
