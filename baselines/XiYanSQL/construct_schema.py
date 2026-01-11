from sqlalchemy import create_engine
from schema_engine import SchemaEngine
import os, json
from tqdm import tqdm

with open("data/bird_minidev_corrections.json") as f:
    corrections = json.load(f)

db_ids = list(set([correction['db_id'] for correction in corrections]))
base_path = "./data/dev/dev_databases"

schemas = {}

for db_name in tqdm(db_ids):
    db_path = f"{base_path}/{db_name}/{db_name}.sqlite"
    abs_path = os.path.abspath(db_path)
    db_engine = create_engine(f'sqlite:///{abs_path}')
    schema_engine = SchemaEngine(engine=db_engine, db_name=db_name)
    mschema = schema_engine.mschema
    mschema_str = mschema.to_mschema()
    dialect = db_engine.dialect.name
    schemas[db_name] = {"schema": mschema_str, "dialect": dialect}

with open("baselines/XiYanSQL-QwenCoder/schemas.json", "w") as f:
    json.dump(schemas, f, indent=2, ensure_ascii=False)
    