#! /bin/sh
# Copyright (C) 2005-2014 Free Software Foundation, Inc.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2, or (at your option)
# any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# Make sure aclocal does not fail on configure.ac inclusions that do
# more than just variable definitions.
# Report from Peter Breitenlohner (PR/450).

. test-init.sh

cat >configure.ac <<END
AC_INIT([$me], [1.0])
m4_include([aconfig.ac])
FOO
AC_OUTPUT
END

cat >aconfig.ac <<'END'
AM_INIT_AUTOMAKE
AC_DEFUN([FOO], [echo GREPME])
sinclude([bconfig.ac])
END

cat >bconfig.ac <<'END'
AC_ARG_WITH([grepme], [string])
END

$ACLOCAL
$AUTOCONF
./configure >stdout || { cat stdout; exit 1; }
cat stdout
grep GREPME stdout
grep 'aconfig\.ac' aclocal.m4 && exit 1
grep 'bconfig\.ac' aclocal.m4 && exit 1
grep with-grepme configure

:
