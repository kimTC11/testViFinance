## Benchmarks
We follow preivous work [1,2] and use the following bugs in Defects4J.

| Identifier      | Project name               |         Bug ids       |
|-----------------|----------------------------|-----------------------|
| Chart           | jfreechart                 |           1-26          |
| Cli             | commons-cli                |           1,2-4,6-11,13-16,21,23-28,30-38,40          |
| Csv             | commons-csv                |           1-2,4-16          |
| Gson            | gson                       |           1-18          |
| Lang            | commons-lang               |           1,3-24,26-41,43-61,63-64          | 

The hash of these bugs can be found in https://github.com/rjust/defects4j/tree/master/framework/projects

## Sampled dev set
The following bugs are sampled dev set

| Project name               |         Bug ids       |
|----------------------------|-----------------------|
| Chart                      |           6          |
| Chart                      |           21          |
| Cli                      |           30          |
| Cli                      |           37          |
| Csv                      |           4          |
| Csv                      |           16          |
| Gson                      |           12          |
| Lang                      |           21          |
| Lang                      |           34          |
| Lang                      |           35          |






[1] Tufano M, Drain D, Svyatkovskiy A, et al. Unit test case generation with transformers and focal context[J]. arXiv preprint arXiv:2009.05617, 2020.  
[2] Alagarsamy S, Tantithamthavorn C, Aleti A. A3test: Assertion-augmented automated test case generation[J]. Information and Software Technology, 2024, 176: 107565.  
