from sqlalchemy import create_engine
from schema_engine import SchemaEngine
import os, json
from tqdm import tqdm

with open("../../spider_2.json") as f:
    corrections = json.load(f)

db_ids = list(set([correction['db_id'] for correction in corrections]))
# base_path = "../../spider_data/test_database"
base_path = "../../spider2_dbs"

schemas = {}

for db_name in tqdm(db_ids):
    db_path = f"{base_path}/{db_name}/{db_name}.sqlite"
    assert os.path.exists(db_path)
    abs_path = os.path.abspath(db_path)
    print(abs_path)
    db_engine = create_engine(f'sqlite:///{abs_path}')
    schema_engine = SchemaEngine(engine=db_engine, db_name=db_name)
    mschema = schema_engine.mschema
    mschema_str = mschema.to_mschema()
    dialect = db_engine.dialect.name
    schemas[db_name] = {"schema": mschema_str, "dialect": dialect}

with open("./spider2_schemas.json", "w") as f:
    json.dump(schemas, f, indent=2, ensure_ascii=False)
    