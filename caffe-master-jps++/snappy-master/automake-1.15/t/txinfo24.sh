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

# Check that info files are built in builddir when needed.
# (Similar to txinfo16.sh, plus CLEANFILES).
# (See also txinfo23.sh and txinfo25.sh).

required='makeinfo tex texi2dvi'
. test-init.sh

cat >> configure.ac << 'END'
AC_OUTPUT
END

cat > Makefile.am << 'END'
CLEANFILES = main.info
info_TEXINFOS = main.texi
END

cat > main.texi << 'END'
\input texinfo
@setfilename main.info
@settitle main
@node Top
Hello walls.
@include version.texi
@bye
END

$ACLOCAL
$AUTOMAKE --add-missing -Wno-obsolete
$AUTOCONF

mkdir build
cd build
../configure
$MAKE
test ! -e ../main.info
test -f main.info

cd ..
rm -rf build
./configure
$MAKE
test -f main.info

# Make sure stamp-vti is older that version.texi.
# (A common situation in a real tree).
test -f stamp-vti
test -f version.texi
$sleep
touch stamp-vti

$MAKE distclean
test -f stamp-vti
test -f version.texi

mkdir build
cd build
../configure
$MAKE
# main.info should be rebuilt in the current directory.
test -f main.info
test ! -e ../main.info
$MAKE dvi
test -f main.dvi

$MAKE distcheck

:
