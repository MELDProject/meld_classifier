Contributing guide
==================

Contributing to meld_classifier
-------------------------------

We use a pull-request based workflow. Do not push to master directly, to avoid having buggy, untested code.
The general workflow is as follows:

1. create a branch for your feature: 
	```
	git checkout -b cool-new-feature
	```
2. push all changes to this branch, test on hpc, etc. 
	- For pushing the branch to the origin use: 
		```
		git push origin cool-new-feature
		```
	- For pulling the branch from the remote (when it does not yet exist locally): 
		```
		git checkout -b cool-new-feature origin/cool-new-feature
		```
4. test your changes by adding unit tests to `meld_classifier/test`. Format your code using `black`.
5. make sure that all tests run by running:
	```
	pytest
	```
	There are some long tests, if you want to exlude them for quick testing, run `pytest -m "not slow"`. 
3. when finished developing, merge in new changes from master: (on `cool-new-feature` branch) 
	```
	git pull origin master
	```
4. create a pull request on github and document all changes that you have done (also useful for looking at in the future). E.g. look at this example #3. Creating a PR works by going to your branch on the github webpage and pressing the "pull request" button. 
5. squash & merge the PR into master. This replaces all the individual commits with 1 commit per PR which is helful for keeping a clean commit history on master and get rid of all the "bugfix" commit messages. Go to the PR on the github webpage and click on the down arrow on the merge button. Select `squash and merge` and click. 


Creating a release
-----------------

We use `bump2version` (``pip install bump2version``) to keep track of version numbering. 
Version numbers follow the convention of ``major.minor.patch``.

1. Create a new branch for this release:

    git checkout -b release/VERSION

2. Update the current version and create a new version tag:

    bump2version {major,minor,patch}
    git push --tags

3. Create and merge a PR for the bump2version commit.

4. Edit and publish the release on github (under the "tags" Tab) and add release notes.
