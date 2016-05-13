#include "utils/db.hpp"
#include "utils/db_lmdb.hpp"

DB* GetDB(const string& backend){
	if (backend == "leveldb"){
		NOT_IMPLEMENTED;
	}
	if (backend == "lmdb"){
		return new LMDB();
	}
	return new LMDB();
}
DB* GetDB(const int backend){
	if (backend == 0){
		NOT_IMPLEMENTED;
	}
	if (backend == 1){
		return new LMDB();
	}
	return new LMDB();
}