# List the compute backends reachable through the OriginQ cloud access channel.
#
# The API token is NEVER hard-coded here.  It is read from $ORIGINQ_TOKEN, or
# failing that from the untracked local file .originq_token -- the same
# convention qcloud_vscr_new.py, qcloud_auto.py and qcloud_run2.py already use.
# .originq_token is in .gitignore; keep it that way.
import os
import sys

from qpanda3_runtime import RuntimeService

ROOT = os.path.dirname(os.path.abspath(__file__))
TOKEN_FILE = os.path.join(ROOT, '.originq_token')


def resolve_token():
    """$ORIGINQ_TOKEN first, then ./.originq_token, else fail loudly."""
    tok = os.environ.get('ORIGINQ_TOKEN', '').strip()
    if tok:
        return tok
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE) as fh:
            tok = fh.read().strip()
        if tok:
            return tok
    sys.exit('no API token: set ORIGINQ_TOKEN or create %s' % TOKEN_FILE)


service = RuntimeService()
service.login(resolve_token())

# 使用RuntimeService.list_devices查询访问通道支持的计算后端设备
devs = service.list_devices()
print('devs:', devs)
