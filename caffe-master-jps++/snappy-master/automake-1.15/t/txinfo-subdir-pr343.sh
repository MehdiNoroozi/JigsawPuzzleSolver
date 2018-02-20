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

# Check for subdir Texinfo (PR/343).

required='makeinfo tex texi2dvi'
. test-init.sh

echo AC_OUTPUT >> configure.ac

cat > Makefile.am << 'END'
info_TEXINFOS = subdir/main.texi
subdir_main_TEXINFOS = subdir/inc.texi

installcheck-local:
	test -f "$(infodir)/main.info"
END

mkdir subdir

cat > subdir/main.texi << 'END'
\input texinfo
@setfilename main.info
@settitle main
@node Top
Hello walls.
@include version.texi
@include inc.texi
@bye
END

cat > subdir/inc.texi << 'END'
I'm included.
END

$ACLOCAL
$AUTOMAKE --add-missing
$AUTOCONF

mkdir build
cd build
../configure
$MAKE distcheck
test -f ../subdir/main.info
test ! -e subdir/main.info

:
