# Requirements

A staging CLI server for adhoc and batch generation requests.

- The server warms up and loads the model files, build the necessary in-memory model instances, so that the incoming generation requests can immediately kick off.
- The server supports in coming generation requests in multiple forms, including but not limiting to
  - adhoc calls with the same parameter interfaces as the `ZImageCLI`; so the users can simply copy-paste the historical `ZImageCLI` commands, change the program to call, and kick off.
  - batch generation requests in structured json file
  - markdown files contain multiple multi-line "\```bash [...] ``` \" blocks, each block contains a ready-to-run CLI command to the existing `ZImageCLI`.
