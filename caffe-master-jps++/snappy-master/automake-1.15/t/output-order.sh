#! /bin/sh
# Copyright (C) 2008-2014 Free Software Foundation, Inc.
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

# Test that 'automake -a' output order is stable.
# From report by Bruno Haible.

. test-init.sh

cat >>configure.ac <<'END'
AC_OUTPUT
END

: >Makefile.am
: >AUTHORS
: >ChangeLog
: >NEWS
: >README

cat >.autom4te.cfg <<'END'
begin-language: "Autoconf"
args: --no-cache
end-language: "Autoconf"
begin-language: "Autoconf-without-aclocal-m4"
args: --no-cache
end-language: "Autoconf-without-aclocal-m4"
END

$ACLOCAL
$AUTOCONF
rm -f missing install-sh
$AUTOMAKE --add-missing --copy 2>stderr || { cat stderr >&2; exit 1; }
cat stderr >&2

for i in 1 2 3 4 5 6; do
  rm -f missing install-sh INSTALL COPYING
  # The grep prevents a Heisenbug with the HP-UX shell and VERBOSE=yes.
  $AUTOMAKE --add-missing --copy 2>&1 >/dev/null |
  grep -v /dev/null |
  diff - stderr
done
