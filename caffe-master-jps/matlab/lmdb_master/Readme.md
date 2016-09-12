Matlab LMDB
===========

Matlab LMDB wrapper for UNIX environment.

 * See [LMDB website](http://symas.com/mdb/).
 * The implementation is based on [mexplus](http://github.com/kyamagu/mexplus).

See also [matlab-leveldb](http://github.com/kyamagu/matlab-leveldb).

The package does not contain any data serialization. Use `char` for storing keys and values. Those using [Caffe](https://github.com/BVLC/caffe) might want to use a [Datum converter](https://gist.github.com/kyamagu/31a4b6f782670a28098b) also.

Build
-----

    addpath /path/to/matlab-lmdb;
    lmdb.make();
    lmdb.make('test');

Edit `Makefile` to customize the build process.

Example
-------

    % Open and close.
    database = lmdb.DB('./db', 'MAPSIZE', 1024^3);
    clear database;
    readonly_database = lmdb.DB('./db', 'RDONLY', true, 'NOLOCK', true);
    clear readonly_database;

    % Read and write.
    database.put('key1', 'value1');
    database.put('key2', 'value2');
    value1 = database.get('key1');
    database.remove('key1');

    % Iterator.
    database.each(@(key, value) disp([key, ': ', value]));
    count = database.reduce(@(key, value, count) count + 1, 0);

    % Cursor.
    cursor = database.cursor('RDONLY', true);
    while cursor.next()
      key = cursor.key;
      value = cursor.value;
    end
    clear cursor;

    % Transaction.
    transaction = database.begin();
    try
      transaction.put('key1', 'value1');
      transaction.put('key2', 'value2');
      transaction.commit();
    catch exception
      transaction.abort();
    end
    clear transaction;

    % Dump.
    keys = database.keys();
    values = database.values();

TODO
----

 * Finer transaction API.
