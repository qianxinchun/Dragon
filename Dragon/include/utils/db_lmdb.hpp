#ifndef DB_LMDB_HPP
#define DB_LMDB_HPP

#include <string>
#include <lmdb/lmdb.h>
#include "../common.hpp"
#include "db.hpp"

inline void MDB_CHECK(int mdb_status){ CHECK_EQ(mdb_status, MDB_SUCCESS) << mdb_strerror(mdb_status); }
class LMDBCursor :public Cursor{
public:
	LMDBCursor(MDB_txn *txn, MDB_cursor *cursor) :
		mdb_txn(txn), mdb_cursor(cursor), valid_(false) {SeekToFirst(); }
	virtual ~LMDBCursor(){
		mdb_cursor_close(mdb_cursor);
		mdb_txn_abort(mdb_txn);
	}
	virtual void SeekToFirst(){ Seek(MDB_FIRST); }
	virtual void Next() { Seek(MDB_NEXT); }
	virtual string key(){
		return string((const char*)mdb_key.mv_data, mdb_key.mv_size);
	}
	virtual string value(){
		return string((const char*)mdb_val.mv_data, mdb_val.mv_size);
	}
	virtual bool valid() { return valid_; }
private:
	void Seek(MDB_cursor_op op){
		int mdb_status = mdb_cursor_get(mdb_cursor, &mdb_key, &mdb_val, op);
		if (mdb_status == MDB_NOTFOUND) valid_ = false;
		else{ MDB_CHECK(mdb_status); valid_ = true; }
	}
	MDB_txn* mdb_txn;
	MDB_cursor* mdb_cursor;
	MDB_val mdb_key, mdb_val;
	bool valid_;
};

class LMDBTransaction : public Transaction{
public:
	LMDBTransaction(MDB_dbi *dbi,MDB_txn *txn):mdb_dbi(dbi), mdb_txn(txn) {}
	virtual void Put(const string& key, const string&val);
	virtual void Commit() { MDB_CHECK(mdb_txn_commit(mdb_txn)); } //封锁Buffer，写入到文件
private:
	MDB_dbi* mdb_dbi;
	MDB_txn* mdb_txn;
};

class LMDB :public DB{
public:
	LMDB() :mdb_env(NULL) {}
	virtual ~LMDB() { Close(); }
	virtual void Open(const string& source, Mode mode);
	virtual void Close(){
		if (mdb_env != NULL){
			mdb_dbi_close(mdb_env, mdb_dbi);
			mdb_env_close(mdb_env);
			mdb_env = NULL;
		}
	}
	virtual LMDBCursor* NewCursor();
	virtual LMDBTransaction* NewTransaction();
private:
	MDB_env* mdb_env;
	MDB_dbi  mdb_dbi;
};


#endif