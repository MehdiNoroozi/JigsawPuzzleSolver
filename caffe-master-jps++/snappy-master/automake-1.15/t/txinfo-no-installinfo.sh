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

# Make sure install-info works even if no-installinfo is given.

required='makeinfo'
. test-init.sh

echo AC_OUTPUT >> configure.ac

cat > Makefile.am << 'END'
info_TEXINFOS = main.texi
AUTOMAKE_OPTIONS = no-installinfo
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
$AUTOMAKE --add-missing
$AUTOCONF

./configure --prefix="$(pwd)/inst" --infodir="$(pwd)/inst/info"
$MAKE install-info
test -f inst/info/main.info

:
