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

# Make sure COPYING is not overwritten, even with -a -f.

. test-init.sh

echo AC_OUTPUT >>configure.ac

cat >Makefile.am <<\EOF
test1: distdir
	grep 'GNU GENERAL PUBLIC LICENSE' $(distdir)/COPYING
test2: distdir
	grep 'MY-OWN-LICENSE' $(distdir)/COPYING
test3: distdir
	test ! -f $(distdir)/COPYING
	grep 'MY-OWN-LICENSE' $(distdir)/COPYING.LIB
EOF

:> NEWS
:> AUTHORS
:> ChangeLog
:> README

test ! -e COPYING

$ACLOCAL
$AUTOCONF
$AUTOMAKE --gnu --add-missing

./configure
$MAKE test1

# Use 'rm' before 'echo', because COPYING is likely to be a symlink to
# the real COPYING...
rm -f COPYING
echo 'MY-OWN-LICENSE' >COPYING
$MAKE test2

$AUTOMAKE --gnu --add-missing --force-missing
./configure
$MAKE test2

rm -f COPYING
echo 'MY-OWN-LICENSE' >COPYING.LIB
$AUTOMAKE --gnu --add-missing --force-missing
./configure
$MAKE test3
