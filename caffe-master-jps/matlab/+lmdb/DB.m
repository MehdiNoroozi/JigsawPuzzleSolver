classdef DB < handle
%DB LMDB wrapper.
%
% % Open and close.
% database = lmdb.DB('./db');
% clear database;
%
% % Read and write.
% database.put('key1', 'value1');
% database.put('key2', 'value2');
% value1 = database.get('key1');
% database.remove('key1');
%
% % Iterator.
% database.each(@(key, value) disp([key, ':', 'value']));
% count = database.reduce(@(key, value, count) count + 1, 0);
%
% % Cursor.
% cursor = database.cursor('RDONLY', true);
% while cursor.next()
%   key = cursor.key;
%   value = cursor.value;
% end
% clear cursor;
%
% See also lmdb.DB.DB

properties (Access = private)
  id_ % ID of the session.
end

methods
  function this = DB(filename, varargin)
  %DB Create a new database.
  %
  % database = lmdb.DB('./db')
  % database = lmdb.DB('./db', 'RDONLY', true, ...)
  %
  % Options
  %   'MODE' default 0664
  %   'FIXEDMAP' default false
  %   'NOSUBDIR' default false
  %   'NOSYNC'  default false
  %   'RDONLY', default false
  %   'NOMETASYNC' default false
  %   'WRITEMAP'  default false
  %   'MAPASYNC' default false
  %   'NOTLS' default false
  %   'NOLOCK' default false
  %   'NORDAHEAD' default false
  %   'NOMEMINIT' default false
  %   'REVERSEKEY'  default false
  %   'DUPSORT' default false
  %   'INTEGERKEY' default false
  %   'DUPFIXED'  default false
  %   'INTEGERDUP'  default false
  %   'REVERSEDUP'  default false
  %   'CREATE'  default true unless 'RDONLY' specified
  %   'MAPSIZE' default 10485760
  %   'MAXREADERS' default 126
  %   'MAXDBS' default 0
  %   'NAME' default ''
    assert(isscalar(this));
    assert(ischar(filename));
    this.id_ = LMDB_('new', filename, varargin{:});
  end

  function delete(this)
  %DELETE Destructor.
    assert(isscalar(this));
    LMDB_('delete', this.id_);
  end

  function result = get(this, key)
  %GET Query a record.
    assert(isscalar(this));
    result = LMDB_('get', this.id_, key);
  end

  function put(this, key, value, varargin)
  %PUT Save a record in the database.
  %
  % Options
  %   'NODUPDATA' default false
  %   'NOOVERWRITE' default false
  %   'RESERVE' default false
  %   'APPEND' default false
    assert(isscalar(this));
    LMDB_('put', this.id_, key, value, varargin{:});
  end

  function remove(this, key, varargin)
  %REMOVE Remove a record.
    assert(isscalar(this));
    LMDB_('remove', this.id_, key, varargin{:});
  end

  function each(this, func)
  %EACH Apply a function to each record.
  %
  % Example: show each record.
  %
  % database.each(@(key, value) disp([key, ': ', value]))
    assert(isscalar(this));
    assert(ischar(func) || isa(func, 'function_handle'));
    assert(abs(nargin(func)) > 1);
    LMDB_('each', this.id_, func);
  end

  function result = reduce(this, func, initial_value)
  %REDUCE Apply an accumulation function to each record.
  %
  % Example: counting the number of 'foo' in the records.
  %
  % database.reduce(@(key, val, accum) accum + strcmp(val, 'foo'), 0)
    assert(isscalar(this));
    assert(ischar(func) || isa(func, 'function_handle'));
    assert(abs(nargin(func)) > 2 && abs(nargout(func)) > 0);
    result = LMDB_('reduce', this.id_, func, initial_value);
  end

  function transaction = begin(this, varargin)
  %TRANSACTION Create a new transaction.
  %
  % transaction = database.begin()
  % try
  %   transaction.put(key, value)
  %   transaction.commit()
  % catch exception
  %   transaction.abort()
  % end
  %
  % Options
  %   'RDONLY' default false
  %
  % Note: Calling a database method when there is an active transaction results
  % in deadlock. Also, destroying a database while there is an active
  % transaction will crash the process.
    assert(isscalar(this));
    transaction = lmdb.Transaction(this.id_, varargin{:});
  end

  function cursor_value = cursor(this, varargin)
  %CURSOR Create a new cursor.
  %
  % Options
  %    'RDONLY' default false
  %
  % Example
  %
  %    cursor = database.cursor('RDONLY', true);
  %    while cursor.next()
  %      key = cursor.key;
  %      value = cursor.value;
  %    end
  %    clear cursor;
  %
  % Clear a cursor before the database.
  %
  % See also lmdb.Cursor
    assert(isscalar(this));
    cursor_value = lmdb.Cursor(this.id_, varargin{:});
  end

  function [key, value] = first(this)
  %FIRST Get the first key-value pair.
  %
  % [key, value] = database.first()
    assert(isscalar(this));
    cursor = this.cursor('RDONLY', true);
    if cursor.next()
      key = cursor.key;
      if nargout > 1
        value = cursor.value;
      end
    else
      key = [];
      value = [];
    end
    clear cursor;
  end

  function result = keys(this)
  %KEYS Get a cell array of all keys.
    assert(isscalar(this));
    result = LMDB_('keys', this.id_);
  end

  function result = values(this)
  %VALUES Get a cell array of all values.
    assert(isscalar(this));
    result = LMDB_('values', this.id_);
  end

  function result = stat(this)
  %STAT Get the environment statistics.
    assert(isscalar(this));
    result = LMDB_('stat', this.id_);
  end
end

end
