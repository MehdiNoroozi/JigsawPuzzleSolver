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

# The install rule should honor failures of the install program.
# Some of these are already caught by 'instmany.sh'.

# This is the libtool sister test of 'instfail.sh'.

required='cc libtool libtoolize'
. test-init.sh

cat >>configure.ac <<END
AM_PROG_AR
AC_PROG_LIBTOOL
AC_OUTPUT
END

cat >Makefile.am <<'END'
bin_PROGRAMS = prog1 prog2 prog3
nobase_bin_PROGRAMS = progn1 progn2 progn3
lib_LTLIBRARIES = liblt1.la liblt2.la liblt3.la
nobase_lib_LTLIBRARIES = libltn1.la libltn2.la libltn3.la
unreadable-prog:
	chmod a-r prog1$(EXEEXT)
readable-prog:
	chmod a+r prog1$(EXEEXT)
unreadable-progn:
	chmod a-r progn1$(EXEEXT)
readable-progn:
	chmod a+r progn1$(EXEEXT)
END

for n in 1 2 3; do
  echo "int main () { return 0; }" > prog$n.c
  echo "int main () { return 0; }" > progn$n.c
  echo "int foolt$n () { return 0; }" > liblt$n.c
  echo "int fooltn$n () { return 0; }" > libltn$n.c
done

libtoolize
$ACLOCAL
$AUTOCONF
$AUTOMAKE --add-missing

instdir=$(pwd)/inst || fatal_ "getting current working directory"
./configure --prefix="$instdir"
$MAKE

$MAKE install
$MAKE uninstall

for file in liblt1.la libltn1.la
do
  chmod a-r $file
  test ! -r $file || skip_ "cannot drop file read permissions"
  $MAKE install-exec && exit 1
  chmod u+r $file
done

$MAKE unreadable-prog
$MAKE install-exec && exit 1
$MAKE readable-prog

$MAKE unreadable-progn
$MAKE install-exec && exit 1
$MAKE readable-progn

:
