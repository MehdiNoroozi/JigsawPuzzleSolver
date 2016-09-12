/** LMDB Matlab wrapper.
 */
#include <lmdb.h>
#include <memory>
#include <mexplus.h>

using namespace std;
using namespace mexplus;

#define ERROR(...) \
    mexErrMsgIdAndTxt("lmdb:error", __VA_ARGS__)
#define ASSERT(cond, ...) \
    if (!(cond)) mexErrMsgIdAndTxt("lmdb:error", __VA_ARGS__)
#define OPTIONFLAG(flag, default_value) \
    ((input.get<bool>(#flag, default_value)) ? MDB_##flag : 0)

namespace {

// Record wrapper.
class Record {
public:
  // Create an empty record.
  Record() {
    mdb_val_.mv_size = 0;
    mdb_val_.mv_data = NULL;
  }
  // Create from string.
  Record(const string& data) {
    initialize(data);
  }
  virtual ~Record() {}
  // Initialize with string.
  void initialize(const string& data) {
    data_.assign(data.begin(), data.end());
    mdb_val_.mv_size = data_.size();
    mdb_val_.mv_data = const_cast<char*>(data_.c_str());
  }
  // Sync the buffer.
  void sync() {
    data_.assign(begin(), end());
    mdb_val_.mv_size = data_.size();
    mdb_val_.mv_data = const_cast<char*>(data_.c_str());
  }
  // Get MDB_val pointer.
  MDB_val* get() { return &mdb_val_; }
  // Beginning of iterator.
  const char* begin() const {
    return reinterpret_cast<const char*>(mdb_val_.mv_data);
  }
  // End of iterator.
  const char* end() const { return begin() + mdb_val_.mv_size; }

private:
  // Data buffer.
  string data_;
  // Dumb MDB_val.
  MDB_val mdb_val_;
};

// Database manager.
class Database {
public:
  // Create an empty database environment.
  Database() : env_(NULL) {
    int status = mdb_env_create(&env_);
    ASSERT(status == MDB_SUCCESS, mdb_strerror(status));
  }
  virtual ~Database() { close(); }
  // Open an environment.
  void openEnv(const char* filename, unsigned int flags, mdb_mode_t mode) {
    ASSERT(env_, "MDB_env not created.");
    int status = mdb_env_open(env_, filename, flags, mode);
    ASSERT(status == MDB_SUCCESS, mdb_strerror(status));
  }
  // Open a table.
  void openDBI(MDB_txn* txn, const char* name, unsigned int flags) {
    ASSERT(env_, "MDB_env not opened.");
    int status = mdb_dbi_open(txn, name, flags, &dbi_);
    ASSERT(status == MDB_SUCCESS, mdb_strerror(status));
  }
  // Close both the table and the environment.
  void close() {
    if (env_) {
      mdb_dbi_close(env_, dbi_);
      mdb_env_close(env_);
    }
    env_ = NULL;
  }
  // Set the size of the memory map to use for this environment.
  void setMapsize(size_t mapsize) {
    int status = mdb_env_set_mapsize(env_, mapsize);
    ASSERT(status == MDB_SUCCESS, mdb_strerror(status));
  }
  // Set the maximum number of threads/reader slots for the environment.
  void setMaxReaders(unsigned int readers) {
    int status = mdb_env_set_maxreaders(env_, readers);
    ASSERT(status == MDB_SUCCESS, mdb_strerror(status));
  }
  // Set the maximum number of named databases for the environment.
  void setMaxDBS(MDB_dbi dbs) {
    int status = mdb_env_set_maxdbs(env_, dbs);
    ASSERT(status == MDB_SUCCESS, mdb_strerror(status));
  }
  // Get the raw MDB_env pointer.
  MDB_env* getEnv() { return env_; }
  // Get the raw MDB_dbi pointer.
  MDB_dbi getDBI() { return dbi_; }

private:
  // MDB_env pointer.
  MDB_env* env_;
  // MDB_dbi pointer.
  MDB_dbi dbi_;
};

// Transaction manager.
class Transaction {
public:
  // Create an empty transaction.
  Transaction() : txn_(NULL), database_(NULL) {}
  // Shorthand for constructor-begin.
  Transaction(Database* database, MDB_txn* parent, unsigned int flags) :
      txn_(NULL), database_(NULL) {
    begin(database, parent, flags);
  }
  virtual ~Transaction() { abort(); }
  // Begin the transaction.
  void begin(Database* database, MDB_txn* parent, unsigned int flags) {
    ASSERT(database, "Null pointer exception.");
    ASSERT(database->getEnv(), "Null pointer exception.");
    abort();
    database_ = database;
    int status = mdb_txn_begin(database_->getEnv(), parent, flags, &txn_);
    ASSERT(status == MDB_SUCCESS, mdb_strerror(status));
  }
  // Commit the transaction.
  void commit() {
    if (txn_) {
      int status = mdb_txn_commit(txn_);
      ASSERT(status == MDB_SUCCESS, mdb_strerror(status));
    }
    txn_ = NULL;
    database_ = NULL;
  }
  // Abort the transaction.
  void abort() {
    if (txn_) {
      mdb_txn_abort(txn_);
    }
    txn_ = NULL;
    database_ = NULL;
  }
  // Open database.
  void openDatabase(const string& name, unsigned int flags) {
    database_->openDBI(txn_,
                       (name == "") ? NULL : name.c_str(),
                       flags);
  }
  // Get the specified database record.
  bool getRecord(Record* key, Record* value) {
    int status = mdb_get(txn_, database_->getDBI(), key->get(), value->get());
    ASSERT(status == MDB_SUCCESS || status == MDB_NOTFOUND,
           mdb_strerror(status));
    return status == MDB_SUCCESS;
  }
  // Put a database record.
  void putRecord(Record* key,
                 Record* value,
                 unsigned int flags) {
    int status = mdb_put(txn_,
                         database_->getDBI(),
                         key->get(),
                         value->get(),
                         flags);
    ASSERT(status == MDB_SUCCESS, mdb_strerror(status));
  }
  // Delete the specified database record.
  void removeRecord(Record* key) {
    int status = mdb_del(txn_, database_->getDBI(), key->get(), NULL);
    ASSERT(status == MDB_SUCCESS, mdb_strerror(status));
  }
  // Get the raw transaction pointer.
  MDB_txn* get() { return txn_; }

private:
  // MDB_txn pointer.
  MDB_txn* txn_;
  // Database pointer.
  Database* database_;
};

// Cursor container.
class Cursor {
public:
  Cursor() : cursor_(NULL) {}
  virtual ~Cursor() { close(); }
  // Open the cursor.
  void open(MDB_txn *txn, MDB_dbi dbi) {
    close();
    int status = mdb_cursor_open(txn, dbi, &cursor_);
    ASSERT(status == MDB_SUCCESS, mdb_strerror(status));
  }
  // Close the cursor.
  void close() {
    if (cursor_)
      mdb_cursor_close(cursor_);
    cursor_ = NULL;
  }
  // Apply the cursor operation and get the value.
  bool get(MDB_cursor_op operation) {
    int status = mdb_cursor_get(cursor_, key_.get(), value_.get(), operation);
    ASSERT(status == MDB_SUCCESS || status == MDB_NOTFOUND,
           mdb_strerror(status));
    return status == MDB_SUCCESS;
  }
  // Put the current key and value.
  void put(unsigned int flags) {
    int status = mdb_cursor_put(cursor_, key_.get(), value_.get(), flags);
    ASSERT(status == MDB_SUCCESS, mdb_strerror(status));
  }
  // Delete the current key and value.
  void remove(unsigned int flags) {
    int status = mdb_cursor_del(cursor_, flags);
    ASSERT(status == MDB_SUCCESS, mdb_strerror(status));
  }
  // Get the raw cursor.
  MDB_cursor* get() { return cursor_; }
  // Get the raw record.
  Record* getKey() { return &key_; }
  // Get the raw record.
  Record* getValue() { return &value_; }

private:
  // MDB_cursor pointer.
  MDB_cursor* cursor_;
  // Key.
  Record key_;
  // Value.
  Record value_;
};

// Create a directory.
void createDirectoryIfNotExist(const mxArray* filename) {
  MxArray flag("dir");
  mxArray* prhs[] = {const_cast<mxArray*>(filename),
                     const_cast<mxArray*>(flag.get())};
  mxArray* plhs;
  ASSERT(mexCallMATLAB(1, &plhs, 2, prhs, "exist") == 0,
         "Failed to check a directory.");
  MxArray status(plhs);
  if (!status.to<bool>())
    ASSERT(mexCallMATLAB(0, NULL, 1, prhs, "mkdir") == 0,
           "Failed to create a directory.");
}

} // namespace

namespace mexplus {

// Template specialization of mxArray* to Record.
template <>
void MxArray::to(const mxArray* array, Record* value) {
  ASSERT(value, "Null pointer exception.");
  value->initialize(MxArray(array).to<string>());
}

// Template specialization of Record to mxArray*.
template <>
mxArray* MxArray::from(const Record& value) {
  return MxArray(string(value.begin(), value.end())).release();
}

// Template specialization of MDB_stat to mxArray*.
template <>
mxArray* MxArray::from(const MDB_stat& stat) {
  MxArray value(Struct());
  value.set("psize", stat.ms_psize);
  value.set("depth", stat.ms_depth);
  value.set("branch_pages", stat.ms_branch_pages);
  value.set("leaf_pages", stat.ms_leaf_pages);
  value.set("overflow_pages", stat.ms_overflow_pages);
  value.set("entries", stat.ms_entries);
  return value.release();
}

// Session instance storage.
template class Session<Database>;
template class Session<Transaction>;
template class Session<Cursor>;

} // namespace mexplus

namespace {

MEX_DEFINE(new) (int nlhs, mxArray* plhs[],
                 int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 1, 23, "MODE", "FIXEDMAP", "NOSUBDIR",
      "NOSYNC", "RDONLY", "NOMETASYNC", "WRITEMAP", "MAPASYNC", "NOTLS",
      "NOLOCK", "NORDAHEAD", "NOMEMINIT", "REVERSEKEY", "DUPSORT",
      "INTEGERKEY", "DUPFIXED", "INTEGERDUP", "REVERSEDUP", "CREATE",
      "MAPSIZE", "MAXREADERS", "MAXDBS", "NAME");
  OutputArguments output(nlhs, plhs, 1);
  unique_ptr<Database> database(new Database);
  ASSERT(database.get() != NULL, "Null pointer exception.");
  database->setMapsize(input.get<size_t>("MAPSIZE", 10485760));
  database->setMaxReaders(input.get<unsigned int>("MAXREADERS", 126));
  database->setMaxDBS(input.get<MDB_dbi>("MAXDBS", 0));
  bool read_only = input.get<bool>("RDONLY", false);
  string filename(input.get<string>(0));
  mdb_mode_t mode = input.get<mdb_mode_t>("MODE", 0664);
  unsigned int flags = OPTIONFLAG(RDONLY, false) |
                       OPTIONFLAG(FIXEDMAP, false) |
                       OPTIONFLAG(NOSUBDIR, false) |
                       OPTIONFLAG(NOSYNC, false) |
                       OPTIONFLAG(NOMETASYNC, false) |
                       OPTIONFLAG(WRITEMAP, false) |
                       OPTIONFLAG(MAPASYNC, false) |
                       OPTIONFLAG(NOTLS, false) |
                       OPTIONFLAG(NOLOCK, false) |
                       OPTIONFLAG(NORDAHEAD, false) |
                       OPTIONFLAG(NOMEMINIT, false);
  if (!read_only)
    createDirectoryIfNotExist(input.get(0));
  database->openEnv(filename.c_str(), flags, mode);
  flags = OPTIONFLAG(REVERSEKEY, false) |
          OPTIONFLAG(DUPSORT, false) |
          OPTIONFLAG(INTEGERKEY, false) |
          OPTIONFLAG(DUPFIXED, false) |
          OPTIONFLAG(INTEGERDUP, false) |
          OPTIONFLAG(REVERSEDUP, false) |
          OPTIONFLAG(CREATE, !read_only);
  Transaction transaction(database.get(), NULL, (read_only) ? MDB_RDONLY : 0);
  transaction.openDatabase(input.get<string>("NAME", ""), flags);
  transaction.commit();
  output.set(0, Session<Database>::create(database.release()));
}

MEX_DEFINE(delete) (int nlhs, mxArray* plhs[],
                    int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 1);
  OutputArguments output(nlhs, plhs, 0);
  Session<Database>::destroy(input.get(0));
}

MEX_DEFINE(get) (int nlhs, mxArray* plhs[],
                 int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 2);
  OutputArguments output(nlhs, plhs, 1);
  Database* database = Session<Database>::get(input.get(0));
  Record key = input.get<Record>(1);
  Record value;
  Transaction transaction(database, NULL, MDB_RDONLY);
  transaction.getRecord(&key, &value);
  transaction.commit();
  output.set(0, value);
}

MEX_DEFINE(put) (int nlhs, mxArray* plhs[],
                 int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 3, 4, "NODUPDATA", "NOOVERWRITE", "RESERVE",
      "APPEND");
  OutputArguments output(nlhs, plhs, 0);
  Database* database = Session<Database>::get(input.get(0));
  unsigned int flags = OPTIONFLAG(NODUPDATA, false) |
                       OPTIONFLAG(NOOVERWRITE, false) |
                       OPTIONFLAG(RESERVE, false) |
                       OPTIONFLAG(APPEND, false);
  Record key = input.get<Record>(1);
  Record value = input.get<Record>(2);
  Transaction transaction(database, NULL, 0);
  transaction.putRecord(&key, &value, flags);
  transaction.commit();
}

MEX_DEFINE(remove) (int nlhs, mxArray* plhs[],
                    int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 2);
  OutputArguments output(nlhs, plhs, 0);
  Database* database = Session<Database>::get(input.get(0));
  Record key = input.get<Record>(1);
  Transaction transaction(database, NULL, 0);
  transaction.removeRecord(&key);
  transaction.commit();
}

MEX_DEFINE(each) (int nlhs, mxArray* plhs[],
                  int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 2);
  OutputArguments output(nlhs, plhs, 0);
  Database* database = Session<Database>::get(input.get(0));
  Transaction transaction(database, NULL, MDB_RDONLY);
  Cursor cursor;
  cursor.open(transaction.get(), database->getDBI());
  while (cursor.get(MDB_NEXT)) {
    MxArray key_array(*cursor.getKey());
    MxArray value_array(*cursor.getValue());
    mxArray* prhs[] = {const_cast<mxArray*>(input.get(1)),
                       const_cast<mxArray*>(key_array.get()),
                       const_cast<mxArray*>(value_array.get())};
    ASSERT(mexCallMATLAB(0, NULL, 3, prhs, "feval") == 0, "Callback failure.");
  }
  cursor.close();
  transaction.commit();
}

MEX_DEFINE(reduce) (int nlhs, mxArray* plhs[],
                    int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 3);
  OutputArguments output(nlhs, plhs, 1);
  Database* database = Session<Database>::get(input.get(0));
  MxArray accumulation(input.get(2));
  Transaction transaction(database, NULL, MDB_RDONLY);
  Cursor cursor;
  cursor.open(transaction.get(), database->getDBI());
  while (cursor.get(MDB_NEXT)) {
    MxArray key_array(*cursor.getKey());
    MxArray value_array(*cursor.getValue());
    mxArray* lhs = NULL;
    mxArray* prhs[] = {const_cast<mxArray*>(input.get(1)),
                       const_cast<mxArray*>(key_array.get()),
                       const_cast<mxArray*>(value_array.get()),
                       const_cast<mxArray*>(accumulation.get())};
    ASSERT(mexCallMATLAB(1, &lhs, 4, prhs, "feval") == 0, "Callback failure.");
    accumulation.reset(lhs);
  }
  cursor.close();
  transaction.commit();
  output.set(0, accumulation.release());
}

MEX_DEFINE(txn_new) (int nlhs, mxArray* plhs[],
                     int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 1, 1, "RDONLY");
  OutputArguments output(nlhs, plhs, 1);
  Database* database = Session<Database>::get(input.get(0));
  unsigned int flags = (input.get<bool>("RDONLY", false)) ? MDB_RDONLY : 0;
  output.set(0, Session<Transaction>::create(
      new Transaction(database, NULL, flags)));
}

MEX_DEFINE(txn_delete) (int nlhs, mxArray* plhs[],
                        int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 1);
  OutputArguments output(nlhs, plhs, 0);
  Session<Transaction>::destroy(input.get(0));
}

MEX_DEFINE(txn_commit) (int nlhs, mxArray* plhs[],
                        int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 1);
  OutputArguments output(nlhs, plhs, 0);
  Transaction* transaction = Session<Transaction>::get(input.get(0));
  transaction->commit();
}

MEX_DEFINE(txn_abort) (int nlhs, mxArray* plhs[],
                       int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 1);
  OutputArguments output(nlhs, plhs, 0);
  Transaction* transaction = Session<Transaction>::get(input.get(0));
  transaction->abort();
}

MEX_DEFINE(txn_get) (int nlhs, mxArray* plhs[],
                     int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 2);
  OutputArguments output(nlhs, plhs, 1);
  Transaction* transaction = Session<Transaction>::get(input.get(0));
  Record key = input.get<Record>(1);
  Record value;
  transaction->getRecord(&key, &value);
  output.set(0, value);
}

MEX_DEFINE(txn_put) (int nlhs, mxArray* plhs[],
                     int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 3, 4, "NODUPDATA", "NOOVERWRITE", "RESERVE",
      "APPEND");
  OutputArguments output(nlhs, plhs, 0);
  Transaction* transaction = Session<Transaction>::get(input.get(0));
  unsigned int flags = OPTIONFLAG(NODUPDATA, false) |
                       OPTIONFLAG(NOOVERWRITE, false) |
                       OPTIONFLAG(RESERVE, false) |
                       OPTIONFLAG(APPEND, false);
  Record key = input.get<Record>(1);
  Record value = input.get<Record>(2);
  transaction->putRecord(&key, &value, flags);
}

MEX_DEFINE(txn_remove) (int nlhs, mxArray* plhs[],
                        int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 2);
  OutputArguments output(nlhs, plhs, 0);
  Transaction* transaction = Session<Transaction>::get(input.get(0));
  Record key = input.get<Record>(1);
  transaction->removeRecord(&key);
}

MEX_DEFINE(cursor_new) (int nlhs, mxArray* plhs[],
                        int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 2);
  OutputArguments output(nlhs, plhs, 1);
  Transaction* transaction = Session<Transaction>::get(input.get(0));
  Database* database = Session<Database>::get(input.get(1));
  unique_ptr<Cursor> cursor(new Cursor);
  cursor->open(transaction->get(), database->getDBI());
  output.set(0, Session<Cursor>::create(cursor.release()));
}

MEX_DEFINE(cursor_delete) (int nlhs, mxArray* plhs[],
                           int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 1);
  OutputArguments output(nlhs, plhs, 0);
  Session<Cursor>::destroy(input.get(0));
}

MEX_DEFINE(cursor_next) (int nlhs, mxArray* plhs[],
                         int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 1);
  OutputArguments output(nlhs, plhs, 1);
  Cursor* cursor = Session<Cursor>::get(input.get(0));
  output.set(0, cursor->get(MDB_NEXT));
}

MEX_DEFINE(cursor_previous) (int nlhs, mxArray* plhs[],
                             int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 1);
  OutputArguments output(nlhs, plhs, 1);
  Cursor* cursor = Session<Cursor>::get(input.get(0));
  output.set(0, cursor->get(MDB_PREV));
}

MEX_DEFINE(cursor_first) (int nlhs, mxArray* plhs[],
                          int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 1);
  OutputArguments output(nlhs, plhs, 1);
  Cursor* cursor = Session<Cursor>::get(input.get(0));
  output.set(0, cursor->get(MDB_FIRST));
}

MEX_DEFINE(cursor_last) (int nlhs, mxArray* plhs[],
                         int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 1);
  OutputArguments output(nlhs, plhs, 1);
  Cursor* cursor = Session<Cursor>::get(input.get(0));
  output.set(0, cursor->get(MDB_LAST));
}

MEX_DEFINE(cursor_find) (int nlhs, mxArray* plhs[],
                         int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 2);
  OutputArguments output(nlhs, plhs, 1);
  Cursor* cursor = Session<Cursor>::get(input.get(0));
  input.get<Record>(1, cursor->getKey());
  output.set(0, cursor->get(MDB_SET));
}

MEX_DEFINE(cursor_getkey) (int nlhs, mxArray* plhs[],
                           int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 1);
  OutputArguments output(nlhs, plhs, 1);
  Cursor* cursor = Session<Cursor>::get(input.get(0));
  output.set(0, *cursor->getKey());
}

MEX_DEFINE(cursor_setkey) (int nlhs, mxArray* plhs[],
                           int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 2, 7, "CURRENT", "NODUPDATA", "NOOVERWRITE",
      "RESERVE", "APPEND", "APPENDDUP", "MULTIPLE");
  OutputArguments output(nlhs, plhs, 0);
  Cursor* cursor = Session<Cursor>::get(input.get(0));
  input.get<Record>(1, cursor->getKey());
  cursor->getValue()->sync();
  unsigned int flags = OPTIONFLAG(CURRENT, true) |
                       OPTIONFLAG(NODUPDATA, false) |
                       OPTIONFLAG(NOOVERWRITE, false) |
                       OPTIONFLAG(RESERVE, false) |
                       OPTIONFLAG(APPEND, false) |
                       OPTIONFLAG(APPENDDUP, false) |
                       OPTIONFLAG(MULTIPLE, false);
  cursor->put(flags);
}

MEX_DEFINE(cursor_getvalue) (int nlhs, mxArray* plhs[],
                             int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 1);
  OutputArguments output(nlhs, plhs, 1);
  Cursor* cursor = Session<Cursor>::get(input.get(0));
  output.set(0, *cursor->getValue());
}

MEX_DEFINE(cursor_setvalue) (int nlhs, mxArray* plhs[],
                             int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 2, 7, "CURRENT", "NODUPDATA", "NOOVERWRITE",
      "RESERVE", "APPEND", "APPENDDUP", "MULTIPLE");
  OutputArguments output(nlhs, plhs, 0);
  Cursor* cursor = Session<Cursor>::get(input.get(0));
  input.get<Record>(1, cursor->getValue());
  cursor->getKey()->sync();
  unsigned int flags = OPTIONFLAG(CURRENT, true) |
                       OPTIONFLAG(NODUPDATA, false) |
                       OPTIONFLAG(NOOVERWRITE, false) |
                       OPTIONFLAG(RESERVE, false) |
                       OPTIONFLAG(APPEND, false) |
                       OPTIONFLAG(APPENDDUP, false) |
                       OPTIONFLAG(MULTIPLE, false);
  cursor->put(flags);
}

MEX_DEFINE(cursor_remove) (int nlhs, mxArray* plhs[],
                           int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 1);
  OutputArguments output(nlhs, plhs, 0);
  Cursor* cursor = Session<Cursor>::get(input.get(0));
  unsigned int flags = OPTIONFLAG(NODUPDATA, true);
  cursor->remove(flags);
}

MEX_DEFINE(keys) (int nlhs, mxArray* plhs[],
                  int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 1);
  OutputArguments output(nlhs, plhs, 1);
  Database* database = Session<Database>::get(input.get(0));
  Transaction transaction(database, NULL, MDB_RDONLY);
  Cursor cursor;
  cursor.open(transaction.get(), database->getDBI());
  vector<string> key_values;
  while (cursor.get(MDB_NEXT)) {
    Record* key = cursor.getKey();
    key_values.push_back(string(key->begin(), key->end()));
  }
  cursor.close();
  transaction.commit();
  output.set(0, key_values);
}

MEX_DEFINE(values) (int nlhs, mxArray* plhs[],
                    int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 1);
  OutputArguments output(nlhs, plhs, 1);
  Database* database = Session<Database>::get(input.get(0));
  Transaction transaction(database, NULL, MDB_RDONLY);
  Cursor cursor;
  cursor.open(transaction.get(), database->getDBI());
  vector<string> value_values;
  while (cursor.get(MDB_NEXT)) {
    Record* value = cursor.getValue();
    value_values.push_back(string(value->begin(), value->end()));
  }
  cursor.close();
  transaction.commit();
  output.set(0, value_values);
}

MEX_DEFINE(stat) (int nlhs, mxArray* plhs[],
                  int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 1);
  OutputArguments output(nlhs, plhs, 1);
  Database* database = Session<Database>::get(input.get(0));
  MDB_stat stat;
  mdb_env_stat(database->getEnv(), &stat);
  output.set(0, stat);
}

} // namespace

MEX_DISPATCH
