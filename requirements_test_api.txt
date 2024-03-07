wheel
nbresult
colorama
ipdb
ipykernel
yapf
matplotlib
pygeohash
pytest
seaborn
numpy==1.23.5
pandas==1.5.3
scipy==1.10.0
scikit-learn==1.3.1
google-cloud-bigquery
google-cloud-storage
db-dtypes
pyarrow
# Trick to install the version of Tensorflow depending on your processor: darwin == Mac, ARM == M1
tensorflow-macos==2.10.0; sys_platform == 'darwin' and 'ARM' in platform_version # Mac M-chips
tensorflow==2.10.0; sys_platform == 'darwin' and 'ARM' not in platform_version # Mac Intel chips
tensorflow==2.10.0; sys_platform != 'darwin' # Windows & Ubuntu Intel chips

# prevent bq 3+ db-dtypes package error when running training from bq

mlflow==2.1.1
prefect==2.14.9

python-dotenv
psycopg2-binary

# API
fastapi==0.108.0
pytz
uvicorn
# tests
httpx
pytest-asyncio
