#! /bin/sh
# Copyright (C) 2010-2014 Free Software Foundation, Inc.
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

# Make sure that our grand macro 'AM_INIT_AUTOMAKE' add proper text
# to the configure help screen.

. test-init.sh

cat > configure.ac <<END
AC_INIT([$me], [1.0])
AM_INIT_AUTOMAKE
END

$ACLOCAL
$AUTOCONF

./configure --help >stdout || { cat stdout; exit 1; }
cat stdout

grep '^  --program-prefix[= ]' stdout
grep '^  --program-suffix[= ]' stdout
grep '^  --program-transform-name[= ]' stdout

:
