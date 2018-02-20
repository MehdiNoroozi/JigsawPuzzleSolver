#! /bin/sh
# Copyright (C) 1997-2014 Free Software Foundation, Inc.
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

# Make sure non-empty, non-info suffixes are diagnosed.

. test-init.sh

cat > Makefile.am << 'END'
info_TEXINFOS = textutils.texi
END

echo '@setfilename textutils.frob' > textutils.texi
: > texinfo.tex

$ACLOCAL
AUTOMAKE_fails
grep 'textutils\.texi:1:.*textutils\.frob.*extension' stderr

:
