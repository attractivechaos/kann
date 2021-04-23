#!/usr/bin/perl

use strict;
use warnings;

my $gs = ' .:-=+*#%@';
my $width = 28;

while (<>) {
	next if /^#/;
	chomp;
	my @t = split;
	my $label = shift(@t);
	print "===> $label <===\n";
	for (my $i = 0; $i < @t; $i += $width) {
		for (my $j = 0; $j < $width && $j < @t; ++$j) {
			my $x = int($t[$i + $j] * 10);
			print substr($gs, $x, 1);
		}
		print "\n";
	}
}
