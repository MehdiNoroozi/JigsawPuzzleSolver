#! /bin/sh
# Copyright (C) 2007-2014 Free Software Foundation, Inc.
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

# Make sure an AC_CONFIG_FILES, AC_CONFIG_LINKS, and AC_CONFIG_COMMANDS
# are not prerequisites of 'all'.

. test-init.sh

cat >> configure.ac << \END
AC_SUBST([FOO], [foo])
if $create; then
  AC_CONFIG_FILES([file])
  AC_CONFIG_LINKS([link:input])
  AC_CONFIG_COMMANDS([stamp], [echo stamp > stamp])
fi
AC_OUTPUT
END

: >Makefile.am

echo link > input
echo @FOO@ >file.in

$ACLOCAL
$AUTOCONF
$AUTOMAKE

./configure create=false
$MAKE
test ! -e file
test ! -e link
test ! -e stamp

./configure create=:
test -f file
test -f link
test -f stamp
