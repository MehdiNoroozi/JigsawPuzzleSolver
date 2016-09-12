classdef Transaction < handle
%TRANSACTION LMDB transaction wrapper.
%
% transaction = database.begin()
% try
%   transaction.put(key, value)
%   transaction.commit()
% catch exception
%   transaction.abort()
% end
%
% See also lmdb.DB.begin

properties (Access = private)
  id_ % ID of the session.
end

methods (Hidden)
  function this = Transaction(database_id, varargin)
  %TRANSACTION Create a new transaction.
  %
  % See also lmdb.DB.begin
    assert(isscalar(this));
    assert(isscalar(database_id));
    this.id_ = LMDB_('txn_new', database_id, varargin{:});
  end
end

methods
  function delete(this)
  %DELETE Destructor.
    assert(isscalar(this));
    LMDB_('txn_delete', this.id_);
  end

  function commit(this)
  %ABORT Commit a transaction.
    assert(isscalar(this));
    LMDB_('txn_commit', this.id_);
  end

  function abort(this)
  %ABORT Abort a transaction.
    assert(isscalar(this));
    LMDB_('txn_abort', this.id_);
  end

  function result = get(this, key)
  %GET Query a record.
    assert(isscalar(this));
    result = LMDB_('txn_get', this.id_, key);
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
    LMDB_('txn_put', this.id_, key, value, varargin{:});
  end

  function remove(this, key, varargin)
  %REMOVE Remove a record.
    assert(isscalar(this));
    LMDB_('txn_remove', this.id_, key, varargin{:});
  end
end

end
