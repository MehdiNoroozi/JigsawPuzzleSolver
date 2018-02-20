#! /bin/sh
# Copyright (C) 2002-2014 Free Software Foundation, Inc.
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

# Test for a special setup where the package's version isn't defined
# in configure.ac.  We want GNU Make for this test (part of the test
# is to make sure Makefile.ins get rebuilt when a m4_included file
# changes -- we don't support this feature on non-GNU Makes).

required='makeinfo tex texi2dvi'
. test-init.sh

cat >configure.ac <<END
m4_include([version.m4])
AC_INIT([$me], [THE_VERSION])
AM_INIT_AUTOMAKE
AC_CONFIG_FILES([Makefile])
AC_OUTPUT
END

echo 'm4_define([THE_VERSION], [2.718])' > version.m4

cat > Makefile.am << 'END'
info_TEXINFOS = zardoz.texi

check:
	test -f $(srcdir)/version.m4
END

cat > zardoz.texi << 'END'
\input texinfo
@setfilename zardoz.info
@settitle Zardoz
@node Top
Hello walls.
@include version.texi
@bye
END

$ACLOCAL
$AUTOCONF
$AUTOMAKE --add-missing
./configure --version | grep '2\.718'
./configure
$MAKE
grep '2\.718' version.texi

$sleep
echo 'm4_define([THE_VERSION], [3.141])' > version.m4
using_gmake || $MAKE Makefile
$MAKE distcheck
./configure --version | grep '3\.141'
grep '3\.141' version.texi

:
