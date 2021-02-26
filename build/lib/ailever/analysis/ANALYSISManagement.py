def dashboard(name, host='127.0.0.1', port='8050'):
    from ._dashboard import dashboard
    dashboard(name=name, host=host, port=port)
