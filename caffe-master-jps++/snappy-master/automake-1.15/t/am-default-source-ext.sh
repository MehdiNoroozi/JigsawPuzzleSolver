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

# AM_DEFAULT_SOURCE_EXT

required='cc c++'
. test-init.sh

cat >> configure.ac << 'END'
AC_PROG_CC
AC_PROG_CXX
AC_CONFIG_FILES([sub/Makefile sub2/Makefile])
AM_CONDITIONAL([COND], [:])
AC_OUTPUT
END

mkdir sub sub2

cat > Makefile.am << 'END'
SUBDIRS = sub sub2
bin_PROGRAMS = foo
END

cat > sub/Makefile.am << 'END'
bin_PROGRAMS = bar baz
AM_DEFAULT_SOURCE_EXT = .cpp
END

cat > sub2/Makefile.am << 'END'
bin_PROGRAMS = bla
if COND
AM_DEFAULT_SOURCE_EXT = .foo .quux
endif
SUFFIXES = .foo .c
.foo.c:
	cat $< >$@
BUILT_SOURCES = bla.c
CLEANFILES = bla.c
END

cat > foo.c << 'END'
int main () { return 0; }
END

cp foo.c sub/bar.cpp
cp foo.c sub/baz.cpp
cp foo.c sub2/bla.foo

$ACLOCAL
$AUTOCONF

# Conditional AM_DEFAULT_SOURCE_EXT does not work yet  :-(
# (this limitation could be lifted).
AUTOMAKE_fails --add-missing
grep 'defined conditionally' stderr

sed '/^if/d; /^endif/d' sub2/Makefile.am > t
mv -f t sub2/Makefile.am

# AM_DEFAULT_SOURCE_EXT can only assume one value
# (lifting this limitation is not such a good idea).
AUTOMAKE_fails --add-missing
grep 'at most one value' stderr

sed 's/ \.quux//' sub2/Makefile.am > t
mv -f t sub2/Makefile.am

$AUTOMAKE --add-missing

./configure
$MAKE
$MAKE distcheck

:
