# Gravity Lab vendoring record

This directory was imported from
[`rgimad/gravity_defied_cpp`](https://github.com/rgimad/gravity_defied_cpp), commit
`91bd283959b96a7ea07e1c4c0040460334c85458` dated 2025-05-12.

The upstream authors listed in its README are rgimad, AntonEvmenenko, and Max Logaev. The original
game credits in that README are preserved. All files remain under GPL-2.0; see `LICENSE.md`.

Gravity Lab modifications:

- 2026-08-28: changed `FileStream` to inherit publicly from `std::fstream`, fixing destruction and
  compilation with modern Apple libc++.
- 2026-08-28: corrected the filename case of the `Time.h` include for portable filesystems.

The classic executable is currently the faithful human-play/reference implementation. The
top-level `gravity_lab_core` remains a separate deterministic RL sandbox until a versioned adapter
around the classic fixed-point physics is complete.
