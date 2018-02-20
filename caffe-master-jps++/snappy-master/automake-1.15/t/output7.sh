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

# Check for support for colon separated input files in AC_CONFIG_FILES,
# with sources in sub directories.

. test-init.sh

cat > configure.ac << END
AC_INIT([$me], [1.0])
AM_INIT_AUTOMAKE
AC_CONFIG_FILES([
  Makefile:mk/toplevel.in
  sub/Makefile:mk/sub.in
  mk/Makefile
])
AC_OUTPUT
END

mkdir mk sub
cat >mk/Makefile.am <<'EOF'
all-local:
	@echo in--mk
EOF

cat >mk/sub.am <<'EOF'
EXTRA_DIST = foo
all-local:
	@echo in--sub
EOF

cat >mk/toplevel.am <<'EOF'
all-local:
	@echo at--toplevel
SUBDIRS = mk sub
EOF

# We have to distribute something in foo, because some versions
# of tar do not archive empty directories when passed the 'o'
# flags.  (This was fixed in GNU tar 1.12, but older
# versions are still used: NetBSD 1.6.1 ships with tar 1.11.2).
#
# If sub/ is missing from the archive, config.status will fail
# to compute $ac_abs_srcdir during a VPATH build: config.status
# is able to create sub/ in the build tree, but it assumes the
# directory already exists in the source tree.
echo bar > sub/foo

$ACLOCAL
$AUTOCONF
$AUTOMAKE
./configure
run_make -O
grep in--mk stdout
grep in--sub stdout
grep at--toplevel stdout

$MAKE distcheck
