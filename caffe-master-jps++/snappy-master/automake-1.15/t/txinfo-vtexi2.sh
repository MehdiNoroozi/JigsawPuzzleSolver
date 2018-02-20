#! /bin/sh
# Copyright (C) 1996-2014 Free Software Foundation, Inc.
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

# Test for bug reported by Glenn Amerine:
#   When automake sees version.texi is being included by a texi file,
#   version.texi gets listed as a dependency for the .info file but
#   not the .dvi file.

. test-init.sh

cat > Makefile.am << 'END'
info_TEXINFOS = zardoz.texi
END

cat > zardoz.texi << 'END'
@setfilename zardoz.info
@include version.texi
END

# Required when using Texinfo.
: > mdate-sh
: > texinfo.tex

$ACLOCAL
$AUTOMAKE

grep '^zardoz\.dvi:.*[ /]version.texi' Makefile.in

:
