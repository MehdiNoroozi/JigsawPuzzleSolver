#! /bin/sh
# Copyright (C) 2003-2014 Free Software Foundation, Inc.
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

# Make sure we diagnose and fix white spaces following backslash.
# Report from Peter Muir.

. test-init.sh

echo AC_OUTPUT >>configure.ac

# Note: trailing whitespace used during the test should not appear as
# trailing whitespace in this file, or it will get stripped by any
# reasonable editor.

echo 'bin_SCRIPTS = foo \ ' >Makefile.am
cat >>Makefile.am <<'END'
bar
ok:
	:
END
echo 'data_DATA = baz \  ' >>Makefile.am
echo '	fum' >>Makefile.am

$ACLOCAL
$AUTOCONF
AUTOMAKE_fails
grep ':1:.*whitespace' stderr
grep ':5:.*whitespace' stderr
$AUTOMAKE -Wno-error
./configure
# Older versions of Automake used to produce invalid Makefiles such input.
$MAKE ok
