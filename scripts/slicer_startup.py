"""Slicer startup for scripts/launch_slicer.sh: optionally opens a folder in FluoroReg, selects the start
module and starts a localhost-only WebServer (scripted remote control).

Env: FLUOROREG_START_MODULE (module selected at startup; default FluoroReg), FLUOROREG_OPEN (experiment or run
folder, optional), FLUOROREG_SLICER_WEBSERVER_PORT (port, optional).
"""
import os

import slicer


def start_localhost_webserver(port):
    """Slicer's WebServer with exec enabled, bound to 127.0.0.1 only (WebServerLogic.start binds all
    interfaces). Fails if the port is taken instead of silently moving to another one."""
    import WebServer

    logic = WebServer.WebServerLogic(port=port, enableSlicer=True, enableExec=True, enableDICOM=False,
                                     enableStaticPages=False)
    logic.server = WebServer.SlicerHTTPServer(requestHandlers=logic.requestHandlers, docroot=logic.docroot,
                                              server_address=("127.0.0.1", port), logMessage=logic.logMessage,
                                              certfile=None, keyfile=None, enableCORS=False)
    logic.server.start()
    logic.serverStarted = True
    return logic


module = os.environ.get("FLUOROREG_START_MODULE", "") or "FluoroReg"
folder = os.environ.get("FLUOROREG_OPEN", "")
port = os.environ.get("FLUOROREG_SLICER_WEBSERVER_PORT", "")

if folder:
    slicer.util.selectModule("FluoroReg")
    try:
        slicer.modules.fluororeg.widgetRepresentation().self().set_project(folder)
    except Exception as e:  # keep Slicer (and the WebServer) up so the failure can be inspected
        import traceback

        traceback.print_exc()
        print(f"FLUOROREG_OPEN_FAILED {folder}: {e}", flush=True)
slicer.util.selectModule(module)
if port:
    slicer.fluororeg_webserver = start_localhost_webserver(int(port))
print(f"FLUOROREG_SLICER_READY module={module} open={folder or '-'} "
      f"webserver={'127.0.0.1:' + port if port else 'off'}", flush=True)
