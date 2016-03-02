#include "data_include/db_lmdb.hpp"
#include "direct.h" 
//Windows includes "direct.h" for _mkdir(dir)
//Linux includes "sys/stat.h" for mkdir(dir,authority
const size_t LMDB_MAP_SIZE = 1099511627776;		//1 TB
void LMDB::Open(const string& source, Mode mode){
	MDB_CHECK(mdb_env_create(&mdb_env));
	MDB_CHECK(mdb_env_set_mapsize(mdb_env, LMDB_MAP_SIZE));
	if (mode == NEW) CHECK_EQ(_mkdir(source.c_str()), 0);
	int flags = 0;
	if (mode == READ) flags = MDB_RDONLY | MDB_NOTLS;
	int rc = mdb_env_open(mdb_env, source.c_str(), flags, 0664);
#ifndef ALLOW_LMDB_NOLOCK
	MDB_CHECK(rc);
#endif
	if (rc == EACCES){
		printf("Permission denied. Trying with MDB_NOLOCK\n");
		mdb_env_close(mdb_env);
		MDB_CHECK(mdb_env_create(&mdb_env));
		flags |= MDB_NOLOCK;
		MDB_CHECK(mdb_env_open(mdb_env, source.c_str(), flags, 0664));
	}
	else MDB_CHECK(rc);
	printf("Open lmdb file: %s\n", source.c_str());
}

LMDBCursor* LMDB::NewCursor(){
	MDB_txn* txn;
	MDB_cursor* cursor;
	MDB_CHECK(mdb_txn_begin(mdb_env, NULL, MDB_RDONLY, &txn));
	MDB_CHECK(mdb_dbi_open(txn, NULL, 0, &mdb_dbi));
	MDB_CHECK(mdb_cursor_open(txn, mdb_dbi, &cursor));
	return new LMDBCursor(txn, cursor);
}

LMDBTransaction* LMDB::NewTransaction(){
	MDB_txn *txn;
	MDB_CHECK(mdb_txn_begin(mdb_env, NULL, 0, &txn));
	MDB_CHECK(mdb_dbi_open(txn, NULL, 0, &mdb_dbi));
	return new LMDBTransaction(&mdb_dbi, txn);
}

void LMDBTransaction::Put(const string& key, const string& val){
	MDB_val mkey, mval;
	mkey.mv_data = (void*)key.data();
	mkey.mv_size = key.size();
	mval.mv_data = (void*)val.data();
	mval.mv_size = val.size();
	MDB_CHECK(mdb_put(mdb_txn, *mdb_dbi, &mkey, &mval, 0));

}
