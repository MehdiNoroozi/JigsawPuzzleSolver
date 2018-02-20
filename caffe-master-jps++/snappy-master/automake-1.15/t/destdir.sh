#! /bin/sh
# Copyright (C) 2002-2014 Free Software Foundation, Inc.
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

# Make sure that 'make distcheck' can find some $(DESTDIR) omissions.
# PR/186.

# The feature we test here relies on read-only directories.
# It will only work for non-root users.
required='ro-dir'

. test-init.sh

cat >> configure.ac <<'EOF'
AC_OUTPUT
EOF

cat > Makefile.am <<'EOF'
dist_data_DATA = foo

# This rule is bogus because it doesn't use $(DESTDIR) on the
# second argument of cp.  distcheck is expected to catch this.
install-data-hook:
	cp $(DESTDIR)$(datadir)/foo $(datadir)/bar

uninstall-local:
	rm -f $(DESTDIR)$(datadir)/bar
EOF

: > foo

$ACLOCAL
$AUTOCONF
$AUTOMAKE -a
./configure
$MAKE distcheck && exit 1

:
