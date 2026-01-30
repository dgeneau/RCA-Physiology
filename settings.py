import os
from dotenv import load_dotenv
load_dotenv()

SITE_URL = os.environ.get("SITE_URL","https://apps.csipacific.ca")
APP_URL = os.environ.get("APP_URL","http://127.0.0.1:8050")

AUTH_URL = f"{SITE_URL}/o/authorize"
TOKEN_URL = f"{SITE_URL}/o/token/"
CLIENT_ID = 'bDf3z9KwxSzCFtxabQ10UwlnHCMl2IsE5teZWLu4' #os.environ.get("CLIENT_ID")
CLIENT_SECRET = 'em7L8NeqjKP8vxTEYRz7LrnHKz7aU8pm7t0DfbCiyQkljgz2YEyf7j2wCfWuN3m21QfKehzAwkwBc8boXGYSOJWFm6PAif4iHQ3kbT5xZ5safDeBlt03YDgqr5EhooYR' #os.environ.get("CLIENT_SECRET")



SPORT_ORG_ENDPOINT = f"/api/registration/organization/"
PROFILE_ENDPOINT = f"/api/registration/profile/"

RAW_INGEST_ENDPOINT = f"/api/warehouse/ingestion/primary/"

# UUID-only: pass the DataSource UUID string directly
#VO2_STEP_SOURCE_UUID = "os.environ.get("WELLNESS_SOURCE_UUID")

