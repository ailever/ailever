def hbar(LEVELs, title='title', returnTrue=False):
    from ._hbar import visualize
    return visualize(LEVELs=LEVELs, title=title, returnTrue=returnTrue)

def dashboard(name, host='127.0.0.1', port='8050'):
    from ._dashboard import dashboard
    dashboard(name=name, host=host, port=port)
