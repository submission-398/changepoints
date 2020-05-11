## Real-world Experimental Setup
### Subject Systems
* LRZIP is a configurable file compression application. We selected 743 commits from its [Git repository](https://github.com/ckolivas/lrzip)'s master branch and sampled 79 configurations. 
* XZ is a configurable file compression application. We selected 1193 commits from its [Git repository](https://git.tukaani.org/xz.git)'s master branch and sampled 161 configurations.
* OGGENC is a audio codec for the .ogg standard of lossy audio compression. We selected 935 configurations from its [Git repository](https://github.com/xiph/vorbis-tools)'s master branch and sampled 152 configurations.

### Benchmarks
For LRZIP and XZ, we compressed the [Silesia Corpus](http://sun.aei.polsl.pl/~sdeor/index.php?page=silesia), a collection of files of different kinds commonly used to assess different compression algorithms. For OGGENC, we encoded a [WAVE audio file](https://commons.wikimedia.org/wiki/File:%22Going_Home%22,_performed_by_the_United_States_Air_Force_Band.wav) from the Wikimedia Commons collection to the .ogg format.

### Performance Measurements
We provide the performance measurements for the three software systems in the subfolder ``data``. Missing values are either due to (partially) invalid configurations as the configuration interface of a software system evolves, or due to commits that fail to build. Below, you can find plots fo the performance histories of all configurations. We decided to use a log-scale on the y-axis since we employed a relative threshold (10%) to identify performance change.

![Performance Histories for Configurations of LRZIP](images/lrzip.png)
![Performance Histories for Configurations of XZ](images/xz.png)
![Performance Histories for Configurations of OGGENC](images/oggenc.png)

### Manually Identified Change Points
We have identified 7 change points for LRZIP and two for XZ and OGGENC, respectively. Below, we present the correpsonding commit id (see the .csv files for performance measurements), the commit hash, and the commit message.
#### LRZIP

| id  | hex                                      | message                                                                                                                      |
|-----|------------------------------------------|------------------------------------------------------------------------------------------------------------------------------|
| 521 | f792f72aa5086439c266433b39303c6377df909b | Use ffsl for a faster lesser_bitness function.                                                                               |
| 529 | 5edf8471d1eedf29ad2e477653062b3b7ca33c9e | Perform all checksumming in a separate thread to speed up the hash search in the rzip phase.                                 |
| 539 | f8d05b9a66fed0ab9fccc77fa2f99c476e4f8382 | Move zpaq compression to new libzpaq library back end.                                                                       |
| 628 | f378595dcec9bd7fdba88ecfb9818112a2b0887e | Make match_len a function completely removing all indirect calls to get_sb, significantly speeding up the single_get_sb case |
| 669 | 70bd5e9d3add335c67ea9535e6ea41af61edab3e | Allow less than maxram to be malloced for checksum to fix Failed to malloc ckbuf in hash_search2                             |
| 675 | 2086185ed58f8ccb2da7cae8c711195513df0919 | *(merge commit)*                                                                                                                        |
| 683 | 321c80f3822681d42e6ef0ec6d805a20fee1e659 | *(merge commit)*                                                                                                                        |
#### XZ
| id  | hex                                      | message                                                                                                                                                                                                                                                                         |
|-----|------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 626 | 77fe5954cd3d10fb1837372684cbc133b56b6a87 | liblzma: Adjust default depth calculation for HC3 and HC4.      It was 8 + nice_len / 4, now it is 4 + nice_len / 4. This allows faster settings at lower nice_len values, even though it seems626 that I won't use automatic depth calcuation with HC3 and HC4 in the presets. |
| 945 | 5db75054e900fa06ef5ade5f2c21dffdd5d16141 | liblzma: Use lzma_memcmplen() in the match finders.      This doesn't change the match finder output.                                                                                                                                                                           |

#### OGGENC

| id  | hex                                      | message                                                                                                                                                                                                                                                                   |
|-----|------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 699 | 4c9327defebfed7534ff0e27a359004032f1540c | ogg123: Conditionally remove use of obsolete CURLOPT_MUTE option. This option     has been obsolete since 2001 and is not defined in recent libcurl headers.     Patch from qboosh (PLD Linux). Closes ticket:1097          svn path=/trunk/vorbis-tools/; revision=12202 |
| 897 | 514116d7bea89dad9f1deb7617b2277b5e9115cd | oggenc: fix crash on raw file close, reported by Hanno in issue #2009. pointer to a non-static struct was escaping its scope. Also fix a C99-ism.          svn path=/trunk/vorbis-tools/; revision=19117                                                                  |
