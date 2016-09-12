function make(varargin)
%MAKE Build MEX files.
%
% Build MEX files in the project.
%
% Example
%
% lmdb.make
% lmdb.make('test')
% lmdb.make('clean')
%
  root = fileparts(fileparts(mfilename('fullpath')));
  command = sprintf('make -C %s MATLABDIR=%s%s', ...
                    root, ...
                    matlabroot, ...
                    sprintf(' %s', varargin{:}));
  disp(command);
  system(command);
end
