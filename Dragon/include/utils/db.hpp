#ifndef DB_HPP
#define DB_HPP

#include <string>
#include "common.hpp"
using namespace std;

class Cursor{
public:
	Cursor() {}
	virtual ~Cursor() {}
	virtual void SeekToFirst() = 0;
	virtual void Next() = 0;
	virtual string key() = 0;
	virtual string value() = 0;
	virtual bool valid() = 0;
};
class Transaction{
public:
	Transaction() {}
	virtual ~Transaction() {}
	virtual void Put(const string& key, const string& val) = 0;
	virtual void Commit() = 0;
};
class DB{
public:
	enum Mode { NEW, READ, WRITE };
	DB() {}
	virtual ~DB() {}
	virtual void Open(const string& source, Mode mode) = 0;
	virtual void Close() = 0;
	virtual Cursor* NewCursor() = 0;
	virtual Transaction* NewTransaction() = 0;

};
DB* GetDB(const string& backend);
DB* GetDB(const int backend);
#endif