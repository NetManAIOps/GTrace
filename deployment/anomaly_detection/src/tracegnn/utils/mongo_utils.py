from datetime import datetime
from pymongo import MongoClient
from pathlib import Path
from tracegnn.data.trace_graph import TraceGraphIDManager
from pathlib import Path
import gridfs
import os
from loguru import logger

class MongoValue:
    def __init__(self, mongo_url: str, db_name: str, collection_name: str, key: str = None):
        self.client = MongoClient(mongo_url)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]

        if key and collection_name not in self.db.list_collection_names():
            self.collection.create_index(key)

    def set_value(self, value):
        self.collection.insert_one(value)

    def get_value(self, key):
        return self.collection.find_one(key)

    def get_value_count(self):
        return self.collection.estimated_document_count()

    def put_simple_kv(self, key: str, value: str):
        self.collection
        if self.collection.find_one({'key': key}) is not None:
            self.collection.update_one({'key': key}, {'$set': {'value': value}})
        else:
            self.collection.insert_one({'key': key, 'value': value})

    def get_simple_kv(self, key: str):
        result = self.collection.find_one({'key': key})

        if result is None:
            return None
        else:
            return result.get('value')


def put_simple_kv(mongo_url: str, key: str, value: str, collection: str = 'default'):
    with MongoClient(mongo_url) as client:
        col = client['kv_db'][collection]

        # Create index for fast find
        col.create_index('key')

        if col.find_one({'key': key}) is not None:
            col.update_one({'key': key}, {'$set': {'value': value}})
        else:
            col.insert_one({'key': key, 'value': value})


def get_simple_kv(mongo_url: str, key: str, collection: str = 'default'):
    with MongoClient(mongo_url) as client:
        col = client['kv_db'][collection]

        result = col.find_one({'key': key})

        if result is None:
            return None
        else:
            return result.get('value')


def load_mongo_file(mongo_url: str,
                    src_file: str,
                    dst_file: Path,
                    version: str = None,
                    db_name: str = 'gridfs_db'):
    dst_file = Path(dst_file)

    with MongoClient(mongo_url) as client:
        db = client[db_name]

        # Get latest version if version is None
        if version is None:
            version = get_simple_kv(mongo_url, f"{src_file}/latest_version")

        # Find file
        fs = gridfs.GridFS(db)

        if version is None:
            cursor = fs.find_one({'filename': src_file})
        else:
            cursor = fs.find_one({'filename': src_file, 'metadata': {'version': version}})

        if cursor is None:
            raise ValueError(f"{src_file}:{version} does not exist.")

        # Save file
        dst_file.parent.mkdir(parents=True, exist_ok=True)

        with open(dst_file, 'wb') as f:
            f.write(cursor.read())

    return version


def save_mongo_file(mongo_url: str,
                     src_file: str,
                     dst_file: str,
                     version: str = None,
                     db_name: str = 'gridfs_db',
                     rewrite: bool = False):
    with MongoClient(mongo_url) as client:
        db = client[db_name]

        # Version
        if version is None:
            version = datetime.strftime(datetime.utcnow(), '%Y-%m-%d %H:%M:%S')

        # Find file
        fs = gridfs.GridFS(db)
        cursor = fs.find_one({'filename': dst_file, 'metadata': {'version': version}})

        if cursor is not None:
            if not rewrite:
                return
            # Delete old file
            fs.delete(cursor)

        # Load file
        with open(src_file, 'rb') as f:
            fs.put(f.read(), filename=dst_file, metadata={'version': version})

    # Write version
    put_simple_kv(mongo_url, f"{dst_file}/latest_version", version)

    return version


def load_id_manager(mongo_url: str, path: Path=Path('/tmp/data/id_manager')):
    path = Path(path)

    logger.info('Downloading id_manager files from mongodb...')

    # Load mongo file
    load_mongo_file(mongo_url=mongo_url,
                    src_file='operation_id.yml',
                    dst_file=path / 'operation_id.yml',
                    version='latest')
    load_mongo_file(mongo_url=mongo_url,
                    src_file='service_id.yml',
                    dst_file=path / 'service_id.yml',
                    version='latest')
    load_mongo_file(mongo_url=mongo_url,
                    src_file='status_id.yml',
                    dst_file=path / 'status_id.yml',
                    version='latest')

    # Read from file
    id_manager = TraceGraphIDManager(path)

    logger.info('id_manager loaded succesfully.')

    return id_manager

