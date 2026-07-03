#include "internal/object_table.h"
#include <cassert>
#include <cstdio>
int main(){ ObjectTable t; int x[8]; ObjectEntry e; e.path="/tmp/memtier_object_table_test"; e.va={x,sizeof(x)}; e.file_range={ObjectTable::fileIdForPath(e.path),4,16}; int id=t.insert(e); assert(id>0); assert(t.lookupById(id)); assert(t.lookupByPointer(&x[2])); assert(t.lookupById(id)->file_range.file_id==ObjectTable::fileIdForPath(e.path)); assert(t.remove(id)); assert(!t.lookupById(id)); puts("test_object_table passed"); }
