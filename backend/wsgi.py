import os
import sys
from dotenv import load_dotenv

load_dotenv()

try:
    from app import app
except Exception as e:
    import traceback
    print('ERROR importing app:', e, flush=True)
    traceback.print_exc()
    sys.exit(1)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    app.run(host="0.0.0.0", port=port)
