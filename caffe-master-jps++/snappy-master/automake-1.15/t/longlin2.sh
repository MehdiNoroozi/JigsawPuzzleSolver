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

# Long lines should be wrapped.
# Report from Albert Chin.

. test-init.sh

n=1 files= match=
while test $n -le 100
do
  files="$files filename$n"
  match="..........$match"
  n=$(($n + 1))
done
files2=$(echo "$files" | sed s/filename/filenameb/g)

cat >Makefile.am <<EOF
FOO = $files $files2 \
  grepme
EOF

# The 'FOO = ...' line is 2293-byte long.  More than what a POSIX
# conformant system is expected to support.  So do not use grep
# on the non-text file.

# grep $match Makefile.am

$ACLOCAL
$AUTOMAKE

grep $match Makefile.in && exit 1
grep 'filenameb100 grepme' Makefile.in

:
