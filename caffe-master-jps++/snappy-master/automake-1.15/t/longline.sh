#! /bin/sh
# Copyright (C) 2004-2014 Free Software Foundation, Inc.
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

# Long lines of += should be wrapped.
# Report from Simon Josefsson.

. test-init.sh

(echo DUMMY = some_long_filename_1;
for i in 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20;
do
  echo DUMMY += some_long_filename_$i
done) > Makefile.am

$ACLOCAL
$AUTOMAKE
test 80 -ge $(grep DUMMY Makefile.in | wc -c)

:
