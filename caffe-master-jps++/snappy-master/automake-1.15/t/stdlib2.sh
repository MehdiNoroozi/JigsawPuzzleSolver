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

# Check for AM_LDFLAGS = -module
# Report from Kevin P. Fleming.
required=libtool
. test-init.sh

: > README
: > NEWS
: > AUTHORS
: > ChangeLog
: > ltconfig
: > ltmain.sh
: > config.guess
: > config.sub

cat >> configure.ac << 'END'
AC_PROG_CC
AM_PROG_AR
AC_PROG_LIBTOOL
AC_OUTPUT
END

: > Makefile.inc

cat > Makefile.am << 'END'
include Makefile.inc
lib_LTLIBRARIES = nonstandard.la
nonstandard_la_SOURCES = foo.c
FOO = -module
END

$ACLOCAL
AUTOMAKE_fails --add-missing --gnu
grep 'Makefile.am:2:.*nonstandard.la.*standard libtool library name' stderr
grep 'Makefile.am:2:.*libnonstandard.la' stderr

# We will use -Wno-gnu to disable the warning about setting LDFLAGS (below)
# Make sure nonstandard names are diagnosed anyway.
AUTOMAKE_fails --add-missing --gnu -Wno-gnu
grep 'Makefile.am:2:.*nonstandard.la.*standard libtool library name' stderr
grep 'Makefile.am:2:.*libnonstandard.la' stderr

# Make sure nonstandard_la_LDFLAGS is read even if LDFLAGS is used.
cat >Makefile.inc <<'EOF'
LDFLAGS = -lfoo
nonstandard_la_LDFLAGS = $(FOO)
EOF
$AUTOMAKE -Wno-gnu

# Make sure LDFLAGS is read even if nonstandard_la_LDFLAGS is used.
cat >Makefile.inc <<'EOF'
LDFLAGS = $(FOO)
nonstandard_la_LDFLAGS = -lfoo
EOF
$AUTOMAKE -Wno-gnu

# Make sure AM_LDFLAGS is not read if foo_LDFLAGS is used.
cat >Makefile.inc <<'EOF'
nonstandard_la_LDFLAGS = -lfoo
AM_LDFLAGS = -module
EOF
AUTOMAKE_fails
grep 'Makefile.am:2:.*nonstandard.la.*standard libtool library name' stderr
grep 'Makefile.am:2:.*libnonstandard.la' stderr

echo 'AM_LDFLAGS = -module' > Makefile.inc
$AUTOMAKE

# For module, Automake should not suggest the lib prefix.
cat > Makefile.am << 'END'
include Makefile.inc
lib_LTLIBRARIES = nonstandard
nonstandard_SOURCES = foo.c
FOO = -module
END

AUTOMAKE_fails
grep "Makefile.am:2:.*'nonstandard'.*standard libtool module name" stderr
grep 'Makefile.am:2:.*nonstandard.la' stderr
