function testLMDB
%TESTLMDB Test the functionality of LMDB wrapper.

  addpath(fileparts(fileparts(mfilename('fullpath'))));
  % Using a database object.
  try
    test_readonly;
    test_operations;
    test_cursor;
    test_transaction;
    test_datatype;
    test_dump;
  catch exception
    disp(exception.getReport());
  end
  if exist('_testdb', 'dir')
    rmdir('_testdb', 's');
  end
  fprintf('DONE\n');

end

function test_operations
  disp('Testing operations');
  database = lmdb.DB('_testdb');
  value = database.get('some-key');
  assert(isempty(value));
  database.put('some-key', 'foo');
  database.put('another-key', 'bar');
  database.put('yet-another-key', 'baz');
  database.remove('yet-another-key');
  value = database.get('another-key');
  assert(strcmp(value, 'bar'), 'ASSERTION FAILED: value = %s', value);
  database.each(@(key, value) disp([key, ':', value]));
  foo_counter = @(key, value, accum) accum + strcmp(value, 'foo');
  foo_count = database.reduce(foo_counter, 0);
  assert(foo_count == 1, 'Expected %d, but observed %d\n', 1, foo_count);
end

function test_readonly
  disp('Testing read-only');
  database = lmdb.DB('_testdb');
  clear database;
  database = lmdb.DB('_testdb', 'RDONLY', true);
  error_raised = false;
  try
    database.put('key', 'value');
  catch
    error_raised = true;
  end
  assert(error_raised);
end

function test_transaction
  disp('Testing transaction');
  database = lmdb.DB('_testdb');
  % Abort test.
  transaction = database.begin();
  for i = 1:1000
    transaction.put(num2str(i), 'foo');
  end
  transaction.abort();
  clear transaction;
  assert(isempty(database.get('1')));
  % Commit test.
  transaction = database.begin();
  for i = 1:1000
    transaction.put(num2str(i), 'foo');
  end
  transaction.commit();
  assert(strcmp(database.get('1'), 'foo'));
  clear transaction;
  % No transaction.
  database.remove('1');
  assert(isempty(database.get('1')));
  disp(database.stat());
  clear database; % Make sure database is not destroyed before transaction.
end

function test_cursor
  disp('Testing cursor');
  database = lmdb.DB('_testdb');
  database.put('some-key', 'foo');
  database.put('another-key', 'bar');
  database.put('yet-another-key', 'baz');
  cursor = database.cursor();
  i = 1;
  while cursor.next()
    disp([cursor.key, ': ', cursor.value]);
    cursor.value = num2str(i);
    i = i + 1;
  end
  clear cursor;
  cursor = database.cursor('RDONLY', true);
  while cursor.next()
    disp([cursor.key, ': ', cursor.value]);
  end
  assert(cursor.first());
  assert(cursor.last());
  assert(cursor.find('some-key'));
  disp([cursor.key, ': ', cursor.value]);
  clear cursor;
  [key, value] = database.first();
  disp([key, ': ', value]);
  clear database;
end

function test_datatype
  disp('Testing data type');
  database = lmdb.DB('_testdb');
  value = uint8(0:255);
  database.put('1', value);
  value2 = cast(database.get('1'), 'uint8');
  assert(all(value == value2));
  clear database;
end

function test_dump
  disp('Testing dump');
  database = lmdb.DB('_testdb', 'RDONLY', true);
  keys = database.keys;
  values = database.values;
  clear database;
end
