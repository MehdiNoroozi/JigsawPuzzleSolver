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

# Make sure an AC_CONFIG_FILES process filenames with AC_SUBST variables.

. test-init.sh

cat >> configure.ac << \END
AC_SUBST([FOO], [foo])
AC_SUBST([file1], [this.in])
echo @FOO@ >$file1
AC_SUBST([file2], [that])
AC_SUBST([file3], [mumble])
AC_SUBST([file4], [foo])
AC_CONFIG_FILES([this:$file1],, [file1=$file1])
AC_CONFIG_FILES([sub/this:$file1])
AC_CONFIG_FILES([${file2}:this],, [file2=$file2])
AC_CONFIG_FILES([${file3}],, [file3=$file3])
AC_CONFIG_FILES([${file4}:foo.in],, [file4=$file4])
AC_CONFIG_FILES([sub/Makefile])
AC_OUTPUT
END

mkdir sub

cat >Makefile.am <<\EOF
SUBDIRS = sub
DISTCLEANFILES = $(file1)
dist-hook:
	test -f $(distdir)/foo.in
	test ! -f $(distdir)/this
EOF

echo @FOO@ >mumble.in
echo @FOO@ >foo.in
: >sub/Makefile.am

$ACLOCAL
$AUTOCONF
$AUTOMAKE

./configure
$MAKE distcheck
cd sub
rm -f this
$MAKE this
grep foo this
