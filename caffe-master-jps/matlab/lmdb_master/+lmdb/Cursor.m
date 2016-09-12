classdef Cursor < handle
%CURSOR LMDB cursor wrapper.
%
% cursor = database.cursor('RDONLY', true);
% while cursor.next()
%   key = cursor.key;
%   value = cursor.value;
% end
% clear cursor;
%
% See also lmdb.DB.cursor

properties (Access = private)
  id_ % ID of the session.
  transaction_id_ % ID of the transaction session.
  database_id_ % ID of the database session.
end

properties (Dependent)
  key
  value
end

methods (Hidden)
  function this = Cursor(database_id, varargin)
  %CURSOR Create a new cursor.
    assert(isscalar(this));
    assert(isscalar(database_id));
    this.database_id_ = database_id;
    this.transaction_id_ = LMDB_('txn_new', database_id, varargin{:});
    this.id_ = LMDB_('cursor_new', this.transaction_id_, database_id);
  end
end

methods
  function delete(this)
  %DELETE Destructor.
    assert(isscalar(this));
    LMDB_('cursor_delete', this.id_);
    LMDB_('txn_commit', this.transaction_id_);
    LMDB_('txn_delete', this.transaction_id_);
  end

  function flag = next(this)
  %NEXT Proceed to next and return true if the value exists.
    assert(isscalar(this));
    flag = LMDB_('cursor_next', this.id_);
  end

  function flag = previous(this)
  %PREVIOUS Proceed to previous and return true if the value exists.
    assert(isscalar(this));
    flag = LMDB_('cursor_previous', this.id_);
  end

  function flag = first(this)
  %FIRST Proceed to the first and return true if the value exists.
    assert(isscalar(this));
    flag = LMDB_('cursor_first', this.id_);
  end

  function flag = last(this)
  %LAST Proceed to the last and return true if the value exists.
    assert(isscalar(this));
    flag = LMDB_('cursor_last', this.id_);
  end

  function flag = find(this, key)
  %FIND Proceed to the specified key and return true if the value exists.
    assert(isscalar(this));
    flag = LMDB_('cursor_find', this.id_, key);
  end

  function key_value = get.key(this)
  %GETKEY Return the current key.
    key_value = LMDB_('cursor_getkey', this.id_);
  end

  function set.key(this, key_value)
  %SETKEY Set the current key.
    LMDB_('cursor_setkey', this.id_, key_value);
  end

  function setKey(this, key_value, varargin)
  %SETKEY Set the current key.
  %
  % Options
  %    'CURRENT' default true
  %    'NODUPDATA' default false
  %    'NOOVERWRITE' default false
  %    'RESERVE' default false
  %    'APPEND' default false
  %    'APPENDDUP' default false
  %    'MULTIPLE' default false
    LMDB_('cursor_setkey', this.id_, key_value, varargin{:});
  end

  function value_value = get.value(this)
  %GETVALUE Return the current value.
    value_value = LMDB_('cursor_getvalue', this.id_);
  end

  function set.value(this, value_value)
  %SETVALUE Set the current value.
    LMDB_('cursor_setvalue', this.id_, value_value);
  end

  function setValue(this, value_value, varargin)
  %SETVALUE Set the current value.
  %
  % Options
  %    'CURRENT' default true
  %    'NODUPDATA' default false
  %    'NOOVERWRITE' default false
  %    'RESERVE' default false
  %    'APPEND' default false
  %    'APPENDDUP' default false
  %    'MULTIPLE' default false
    LMDB_('cursor_setvalue', this.id_, value_value, varargin{:});
  end

  function remove(this, varargin)
  %REMOVE Delete the current key and value.
  %
  % Options
  %    'NODUPDATA' default false
    LMDB_('cursor_remove', varargin{:});
  end
end

end
