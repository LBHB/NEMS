import subprocess
import os
import datetime
import sys
import logging
import itertools
import json
import socket
import shutil

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, desc, exc, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.automap import automap_base
import pandas.io.sql as psql
import sqlite3

#import nems
from . import get_setting
from . import utils

log = logging.getLogger(__name__)
__ENGINE__ = None


###### Functions for establishing connectivity, starting a session, or
###### referencing a database table

def Engine():
    '''Returns a mysql engine object. Creates the engine if necessary.
    Otherwise returns the existing one.'''
    global __ENGINE__

    uri = _get_db_uri()
    if not __ENGINE__:
        __ENGINE__ = create_engine(uri, pool_recycle=1600)

    return __ENGINE__

    #except Exception as e:
    #    log.exception("Error when attempting to establish a database "
    #                  "connection.", e)
    #    raise(e)


def Session():
    '''Returns a mysql session object.'''
    engine = Engine()
    return sessionmaker(bind=engine)()


def Tables():
    '''Returns a dictionary containing Narf database table objects.'''
    engine = Engine()
    Base = automap_base()
    Base.prepare(engine, reflect=True)
    #'sCellFile': Base.classes.sCellFile,
    #'sBatch': Base.classes.sBatch,
    #'gCellMaster': Base.classes.gCellMaster,
    try:
        tables = {
                'Users': Base.classes.Users,
                'Analysis': Base.classes.Analysis,
                'Batches': Base.classes.Batches,
                'Results': Base.classes.Results,
                'Data': Base.classes.Data,
                'tQueue': Base.classes.tQueue,
                'tComputer': Base.classes.tComputer,
                }
    except:
        tables = {
            'Users': Base.classes.NarfUsers,
            'Analysis': Base.classes.NarfAnalysis,
            'Batches': Base.classes.NarfBatches,
            'Results': Base.classes.Results,
            'Data': Base.classes.NarfData,
            'tQueue': Base.classes.tQueue,
            'tComputer': Base.classes.tComputer,
        }

    sql_engine = get_setting('SQL_ENGINE')
    if sql_engine == 'mysql':
        tables['sBatch'] = Base.classes.sBatch
    return tables


def sqlite_test():

    dbfilepath = os.path.join(get_setting('NEMS_RECORDINGS_DIR'), 'nems0.db')

    conn = sqlite3.connect(dbfilepath)
    sql = "SELECT name FROM sqlite_master WHERE type='table' and name like 'Results'"
    r = conn.execute(sql)
    d = r.fetchone()

    if d is None:
        print("Tables missing, need to reinitialize database?")

        print("Creating db")
        scriptfilename = os.path.join(nems0.NEMS_PATH, 'scripts', 'db', 'nems0.db.sqlite.sql')
        cursor = conn.cursor()

        print("Reading Script...")
        scriptFile = open(scriptfilename, 'r')
        script = scriptFile.read()
        scriptFile.close()

        print("Running Script...")
        cursor.executescript(script)

        conn.commit()
        print("Changes successfully committed")

    conn.close()

    return 1


def _get_db_uri():
    '''Used by Engine() to establish a connection to the database.'''
    #creds = nems_db.util.ensure_env_vars(
    #        ['MYSQL_HOST', 'MYSQL_USER', 'MYSQL_PASS',
    #         'MYSQL_DB', 'MYSQL_PORT', 'SQL_ENGINE',
    #         'NEMS_RECORDINGS_DIR']
    #        )

    sql_engine = get_setting('SQL_ENGINE')
    nems_recording_dir = get_setting('NEMS_RECORDINGS_DIR')
    MYSQL_USER = get_setting('MYSQL_USER')
    MYSQL_PASS = get_setting('MYSQL_PASS')
    MYSQL_DB = get_setting('MYSQL_DB')
    MYSQL_PORT = get_setting('MYSQL_PORT')
    MYSQL_HOST = get_setting('MYSQL_HOST')

    if sql_engine == 'mysql':
        db_uri = 'mysql+pymysql://{0}:{1}@{2}:{3}/{4}'.format(
                MYSQL_USER, MYSQL_PASS, MYSQL_HOST,
                MYSQL_PORT, MYSQL_DB
                )
    elif sql_engine == 'sqlite':
        dbfilepath = os.path.join(nems_recording_dir,'nems0.db')
        if ~os.path.exists(dbfilepath):
            sqlite_test()
        db_uri = 'sqlite:///' + dbfilepath

    return db_uri


def pd_query(sql=None, params=None):
    """
    execture an SQL command and return the results in a dataframe
    params:
        sql: string
            query to execute
            use fprintf formatting, eg
                sql = "SELECT * FROM table WHERE name=%s"
                params = ("Joe")

    TODO : sqlite compatibility?
    """

    if sql is None:
        raise ValueError("parameter sql required")
    engine = Engine()
    sql_engine = get_setting('SQL_ENGINE')
    if sql_engine == 'sqlite':
        # print(sql)
        # print(params)
        if params is not None:
            params = tuple(["'" + p + "'" if type(p) is str else p for p in params])
            sql = sql % params

        d = pd.read_sql_query(sql=sql, con=engine)
    else:
        try:
            d = pd.read_sql_query(sql=text(sql), con=engine.connect(), params=params)
        except exc.SQLAlchemyError as OpErr:
            if OpErr._message().count('Lost connection to MySQL server during query')>0:
                log.warning('Lost connection to MySQL server during query, trying again.')
                d = pd.read_sql_query(sql=sql, con=engine, params=params)
    return d


def sql_command(sql):
    engine = Engine()
    conn = engine.connect()
    conn.execute(sql)

###### Functions that access / manipulate the job queue. #######

def enqueue_models(celllist, batch, modellist, force_rerun=False,
                   user="nems", codeHash="master", jerbQuery='',
                   executable_path=None, script_path=None,
                   priority=1, GPU_job=0, reserve_gb=0):
    """Call enqueue_single_model for every combination of cellid and modelname
    contained in the user's selections.

    for each cellid in celllist and modelname in modellist, will create jobs
    that execute this command on a cluster machine:

    <executable_path> <script_path> <cellid> <batch> <modelname>

    e.g.:
    /home/nems/anaconda3/bin/python /home/nems/nems/fit_single_model.py \
       TAR010c-18-1 271 ozgf100ch18_dlog_wcg18x1_stp1_fir1x15_lvl1_dexp1_basic

    Arguments:
    ----------
    celllist : list
        List of cellid selections made by user.
    batch : string
        batch number selected by user.
    modellist : list
        List of modelname selections made by user.
    force_rerun : boolean (default=False)
        If true, models will be fit even if a result already exists.
        If false, models with existing results will be skipped.
    user : string (default="nems")
        Typically the login name of the user who owns the job
    codeHash : string (default="master")
        Git hash string identifying a commit for the specific version of the
        code repository that should be used to run the model fit.
        Can also accept the name of a branch.
    jerbQuery : dict
        Dict that will be used by 'jerb find' to locate matching jerbs
    executable_path : string (defaults to nems' python3 executable)
        Path to executable python (or other command line program)
    script_path : string (defaults to nems' copy of nems/nems_fit_single.py)
        First parameter to pass to executable
    GPU_job: int
        Whether or not to run run as a GPU job.

    Returns:
    --------
    (queueids, messages) : list
        Returns a tuple of the tQueue id and results message for each
        job that was either updated in or added to the queue.

    See Also:
    ---------
    Narf_Analysis : enqueue_models_callback

    """

    # some parameter values, mostly for backwards compatibility with other
    # queueing approaches
    if user:
        user = user
    else:
        user = 'None'
    linux_user = 'nems'
    allowqueuemaster = 1
    waitid = 0
    parmstring = ''
    rundataid = 0

    engine = Engine()
    conn = engine.connect()

    if executable_path in [None, 'None', 'NONE', '']:
        executable_path = get_setting('DEFAULT_EXEC_PATH')
    if script_path in [None, 'None', 'NONE', '']:
        script_path = get_setting('DEFAULT_SCRIPT_PATH')

    # ensures parametes are passed as lists since single strings will successfully enqueue garbage
    assert type(celllist) is list
    assert type(modellist) is list

    # Convert to list of tuples b/c product object only useable once.
    combined = [(c, b, m) for c, b, m in
                itertools.product(celllist, [batch], modellist)]

    notes = ['%s/%s/%s' % (c, b, m) for c, b, m in combined]
    commandPrompts = ["%s %s %s %s %s" % (executable_path, script_path,
                                          c, b, m)
                      for c, b, m in combined]

    queueids = []
    messages = []
    for note, commandPrompt, combo in zip(notes, commandPrompts, combined):

        sql = 'SELECT * FROM tQueue WHERE note="' + note +'"'
        r = conn.execute(sql)

        if r.rowcount>0:
            # existing job, figure out what to do with it

            x=r.fetchone()
            queueid = x['id']
            complete = x['complete']
            if force_rerun:
                if complete == 1:
                    message = "Resetting existing queue entry for: %s\n" % note
                    sql = "UPDATE tQueue SET complete=0, killnow=0, progname='{}', GPU_job='{}', user='{}' WHERE id={}".format(
                        commandPrompt, GPU_job, user, queueid)
                    r = conn.execute(sql)

                elif complete == 2:
                    message = "Dead queue entry for: %s exists, resetting.\n" % note
                    sql = "UPDATE tQueue SET complete=0, killnow=0, GPU_job='{}' WHERE id={}".format(GPU_job, queueid)
                    r = conn.execute(sql)

                else:  # complete in [-1, 0] -- already running or queued
                    message = "Incomplete entry for: %s exists, skipping.\n" % note

            else:

                if complete == 1:
                    message = "Completed entry for: %s exists, skipping.\n"  % note
                elif complete == 2:
                    message = "Dead entry for: %s exists, skipping.\n"  % note
                else:  # complete in [-1, 0] -- already running or queued
                    message = "Incomplete entry for: %s exists, skipping.\n" % note

        else:
            sql = f"SELECT * FROM Results WHERE batch={combo[1]} and cellid='{combo[0]}' and modelname='{combo[2]}'"
            rres = conn.execute(sql)
            if (rres.rowcount==0) | force_rerun:
                # new job
                sql = "INSERT INTO tQueue (rundataid,progname,priority," +\
                       "GPU_job,reserve_gb,parmstring,allowqueuemaster,user," +\
                       "linux_user,note,waitid,codehash,queuedate) VALUES"+\
                       " ({},'{}',{}," +\
                       "{},{},'{}',{},'{}'," +\
                       "'{}','{}',{},'{}',NOW())"
                sql = sql.format(rundataid, commandPrompt, priority, GPU_job, reserve_gb,
                                 parmstring, allowqueuemaster, user, linux_user,
                                 note, waitid, codeHash)
                r = conn.execute(sql)
                queueid = r.lastrowid
                message = "Added new entry for: %s.\n"  % note
            else:
                message = "Completed Results entry for: %s exists, skipping.\n"  % note
                queueid = 0

        queueids.append(queueid)
        messages.append(message)

    conn.close()

    return zip(queueids, messages)


def enqueue_single_model(cellid, batch, modelname, user=None,
                         force_rerun=False, codeHash="master", jerbQuery='',
                         executable_path=None, script_path=None,
                         priority=1, GPU_job=0, reserve_gb=0):

    zipped = enqueue_models([cellid], batch, [modelname], force_rerun,
                            user, codeHash, jerbQuery, executable_path,
                            script_path, priority, GPU_job, reserve_gb)

    queueid, message = next(zipped)
    return queueid, message


def add_job_to_queue(args, note, force_rerun=False,
                   user="nems", codeHash="master", jerbQuery='',
                   executable_path=None, script_path=None,
                   priority=1, GPU_job=0, reserve_gb=0):
    """
    Low level interaction with tQueue to run single generic job on cluster

    <executable_path> <script_path> <arg1> <arg2> <arg3> ...

    Arguments:
    ----------
    args: list of system arguments to be passed to script_path
    note: unique id for this job ex: "Pupil job: AMT004b12_p_TOR"
    force_rerun : boolean (default=False)
        If true, models will be fit even if a result already exists.
        If false, models with existing results will be skipped.
    user : string (default="nems")
        Typically the login name of the user who owns the job
    codeHash : string (default="master")
        Git hash string identifying a commit for the specific version of the
        code repository that should be used to run the model fit.
        Can also accept the name of a branch.
    jerbQuery : dict
        Dict that will be used by 'jerb find' to locate matching jerbs
    executable_path : string (defaults to nems' python3 executable)
        Path to executable python (or other command line program)
    script_path : string (defaults to nems' copy of nems/nems_fit_single.py)
        First parameter to pass to executable

    Returns:
    --------
    (queueids, messages) : list
        Returns a tuple of the tQueue id and results message for each
        job that was either updated in or added to the queue.

    See Also:
    ---------
    Narf_Analysis : enqueue_models_callback

    """

    # some parameter values, mostly for backwards compatibility with other
    # queueing approaches
    if user:
        user = user
    else:
        user = 'None'
    linux_user = 'nems'
    allowqueuemaster = 1
    waitid = 0
    parmstring = ''
    rundataid = 0

    engine = Engine()
    conn = engine.connect()

    if executable_path in [None, 'None', 'NONE', '']:
        executable_path = get_setting('DEFAULT_EXEC_PATH')
    if script_path in [None, 'None', 'NONE', '']:
        script_path = get_setting('DEFAULT_SCRIPT_PATH')

    # Unpack args into command prompt
    if len(args) == 0:
        commandPrompt = "{0} {1}".format(executable_path, script_path)
    elif len(args) > 0:
        for i, arg in enumerate(args):
            if i == 0:
                commandPrompt = "{0} {1} {2}".format(executable_path, script_path, arg)
            else:
                commandPrompt += " {}".format(arg)

    queueids = []
    messages = []

    sql = 'SELECT * FROM tQueue WHERE note="' + note +'"'

    r = conn.execute(sql)
    if r.rowcount>0:
        # existing job, figure out what to do with it

        x=r.fetchone()
        queueid = x['id']
        complete = x['complete']
        if force_rerun:
            sql = "UPDATE tQueue SET complete=0, killnow=0, progname='{}', user='{}' WHERE id={}".format(
                commandPrompt, user, queueid)
            if complete == 1:
                message = "Resetting existing queue entry for: %s\n" % note
                r = conn.execute(sql)

            elif complete == 2:
                message = "Dead queue entry for: %s exists, resetting.\n" % note
                r = conn.execute(sql)

            elif complete == 0:
                message = "Updating unstarted entry for: %s.\n" % note
                r = conn.execute(sql)

            else:  # complete in [-1] -- already running
                message = "Incomplete entry for: %s exists, skipping.\n" % note

        else:

            if complete == 1:
                message = "Completed entry for: %s exists, skipping.\n"  % note
            elif complete == 2:
                message = "Dead entry for: %s exists, skipping.\n"  % note
            else:  # complete in [-1, 0] -- already running or queued
                message = "Incomplete entry for: %s exists, skipping.\n" % note

    else:
        # new job
        sql = "INSERT INTO tQueue (rundataid,progname,priority," +\
               "reserve_gb,parmstring,allowqueuemaster,user," +\
               "linux_user,note,waitid,codehash,GPU_job,queuedate) VALUES"+\
               " ({},'{}',{}," +\
               "{},'{}',{},'{}'," +\
               "'{}','{}',{},'{}',{},NOW())"
        sql = sql.format(rundataid, commandPrompt, priority, reserve_gb,
                         parmstring, allowqueuemaster, user, linux_user,
                         note, waitid, codeHash, GPU_job)
        r = conn.execute(sql)
        queueid = r.lastrowid
        message = "Added new entry for: %s.\n"  % note

        queueids.append(queueid)
        messages.append(message)

    conn.close()

    return zip(queueids, messages)


def _add_model_to_queue(commandPrompt, note, user, codeHash, jerbQuery,
                        priority=1, rundataid=0):
    """
    Returns:
    --------
    job : tQueue object instance
        tQueue object with variables assigned inside function based on
        arguments.

    See Also:
    ---------
    Narf_Analysis: dbaddqueuemaster

    """

    # TODO: why is narf version checking for list vs string on prompt and note?
    #       won't they always be a string passed from enqueue function?
    #       or want to be able to add multiple jobs manually from command line?
    #       will need to rewrite with for loop to to add this functionality in
    #       the future if needed.

    tQueue = Tables()['tQueue']
    job = tQueue()

    if user:
        user = user
    else:
        user = 'None'
    linux_user = 'nems'
    allowqueuemaster = 1
    waitid = 0
    dt = str(datetime.datetime.now().replace(microsecond=0))

    job.rundataid = rundataid
    job.progname = commandPrompt
    job.priority = priority
    job.parmstring = ''
    job.queuedate = dt
    job.allowqueuemaster = allowqueuemaster
    job.user = user
    job.linux_user = linux_user
    job.note = note
    job.waitid = waitid
    job.codehash = codeHash

    return job


def update_job_complete(queueid=None):
    """
    mark job queueid complete in tQueue
    svd old-fashioned way of doing it
    """
    if queueid is None:
        if 'QUEUEID' in os.environ:
            queueid = os.environ['QUEUEID']
        else:
            log.warning("queueid not specified or found in os.environ")
            return 0

    engine = Engine()
    conn = engine.connect()
    sql = f"UPDATE tQueue SET complete=1 WHERE id={queueid}"
    r = conn.execute(sql)
    conn.close()

    return r
    """
    # fancy sqlalchemy method?
    session = Session()
    qdata = (
            session.query(tQueue)
            .filter(tQueue.id == queueid)
            .first()
            )
    if not qdata:
        # Something went wrong - either no matching id, no matching note,
        # or mismatch between id and note
        log.info("Invalid query result when checking for queueid & note match")
        log.info("/n for queueid: %s"%queueid)
    else:
        qdata.complete = 1
        session.commit()

    session.close()
    """


def update_job_start(queueid=None):
    """
    in tQueue, mark job as active and progress set to 1
    """
    if queueid is None:
        if 'QUEUEID' in os.environ:
            queueid = os.environ['QUEUEID']
        else:
            log.warning("queueid not specified or found in os.environ")
            return 0

    engine = Engine()
    conn = engine.connect()
    sql = ("UPDATE tQueue SET complete=-1,progress=1 WHERE id={}"
           .format(queueid))
    r = conn.execute(sql)
    conn.close()
    return r


def update_job_pid(pid, queueid=None):
    """
    in tQueue, update the PID
    """
    if queueid is None:
        if 'QUEUEID' in os.environ:
            queueid = os.environ['QUEUEID']
        else:
            log.warning("queueid not specified or found in os.environ")
            return 0

    engine = Engine()
    conn = engine.connect()
    sql = ("UPDATE tQueue SET pid={} WHERE id={}"
           .format(pid, queueid))
    r = conn.execute(sql)
    conn.close()
    return r


def update_startdate(queueid=None):
    """
    in tQueue, update the PID
    """
    if queueid is None:
        if 'QUEUEID' in os.environ:
            queueid = os.environ['QUEUEID']
        else:
            log.warning("queueid not specified or found in os.environ")
            return 0

    engine = Engine()
    conn = engine.connect()
    sql = ("UPDATE tQueue SET startdate=CURRENT_TIMESTAMP() WHERE id={}").format(queueid)
    r = conn.execute(sql)
    conn.close()
    return r


def update_job_pid(pid, queueid=None):
    """
    in tQueue, update the PID
    """
    if queueid is None:
        if 'QUEUEID' in os.environ:
            queueid = os.environ['QUEUEID']
        else:
            log.warning("queueid not specified or found in os.environ")
            return 0

    engine = Engine()
    conn = engine.connect()
    sql = ("UPDATE tQueue SET pid={} WHERE id={}"
           .format(pid, queueid))
    r = conn.execute(sql)
    conn.close()
    return r


def update_job_tick(queueid=None):
    """
    update current machine's load in the cluster db and tick off a step
    of progress in the fit in tQueue
    """
    if queueid is None:
        if 'QUEUEID' in os.environ:
            queueid = os.environ['QUEUEID']
        else:
            log.warning("queueid not specified or found in os.environ")
            return 0

    #qsetload_path = get_setting('QUEUE_TICK_EXTERNAL_CMD')
    #if len(qsetload_path) & os.path.exists(qsetload_path):
    #    result = subprocess.run(qsetload_path, stdout=subprocess.PIPE)
    #    r = result.returncode
    #    if r:
    #        log.warning('Error executing qsetload')
    #        log.warning(result.stdout.decode('utf-8'))

    engine = Engine()
    conn = engine.connect()

    try:
        # update computer load
        l1, l5, l15 = os.getloadavg()
        hostname = socket.gethostname()
        sql = ("UPDATE tComputer set load1={},load5={}+second(now())/6000,"+
               "load15={},pingcount=0 where name='{}'").format(
                       l1, l5, l15, hostname)
        r = conn.execute(sql)
    except:
        pass

    # tick off progress, tell daemon that job is live
    sql = ("UPDATE tQueue SET progress=progress+1 WHERE id={}"
           .format(queueid))
    r = conn.execute(sql)

    conn.close()

    return r


#### Results / performance logging

def save_results(modelspec, preview_file, queueid=None):
    """
    save performance data from modelspec to Results
    pull some information out of the queue table if queueid provided
    """

    session = Session()
    db_tables = Tables()
    tQueue = db_tables['tQueue']
    Users = db_tables['Users']
    # Can't retrieve user info without queueid, so if none was passed
    # use the default blank user info
    if queueid:
        job = (
            session.query(tQueue)
            .filter(tQueue.id == queueid)
            .first()
        )
        username = job.user
        narf_user = (
            session.query(Users)
            .filter(Users.username == username)
            .first()
        )
        labgroup = narf_user.labgroup
    else:
        username = ''
        labgroup = 'SPECIAL_NONE_FLAG'

    results_id = update_results_table(modelspec, preview_file, username, labgroup)

    session.close()

    return results_id


def update_results_table(modelspec, preview=None,
                         username="svd", labgroup="lbhb", public=1):
    """
    Save information about a fit based on modelspec.meta
    :param modelspec: NEMS modelspec
    :param preview: filename of saved results figure (optional)
    :param username: username id for logging
    :param labgroup: labgroup id for logging
    :param public: (True) if True, flag as publicly visible outside of labgroup
    :return: results_id identifier for new/updated entry in Results table
    """
    db_tables = Tables()
    Results = db_tables['Results']
    Batches = db_tables['Batches']
    session = Session()
    results_id = None
    cell_count = 1
    for cellidx in range(cell_count):
        #modelspec.cell_index = cellidx
        cellids = modelspec.meta.get('cellids', [modelspec.meta.get('cellid','CELL')])
        if ('r_test' in modelspec.meta.keys()) and (len(modelspec.meta['r_test'])<len(cellids)):
            cellids=cellids[:len(modelspec.meta['r_test'])]
        for cellid in cellids:
            batch = modelspec.meta['batch']
            modelname = modelspec.meta['modelname']
            r = (
                session.query(Batches)
                .filter(Batches.cellid == cellid)
                .filter(Batches.batch == batch)
                .first()
            )
            if not r:
                # add cell/batch to Data
                log.info("Adding (%s/%d) to Batches", cellid, batch)
                r = Batches()
                r.cellid = cellid
                r.batch = batch
                session.add(r)

            r = (
                session.query(Results)
                .filter(Results.cellid == cellid)
                .filter(Results.batch == batch)
                .filter(Results.modelname == modelname)
                .first()
            )
            collist = ['%s' % (s) for s in Results.__table__.columns]
            attrs = [s.replace('Results.', '') for s in collist]
            removals = [
                'id', 'lastmod'
            ]
            for col in removals:
                attrs.remove(col)

            if not r:
                r = Results()
                new_entry = True
            else:
                new_entry = False

            if preview:
                r.figurefile = preview
            # TODO: This overrides any existing username and labgroup assignment.
            #       Is this the desired behavior?
            r.username = username
            r.public = public
            if not labgroup == 'SPECIAL_NONE_FLAG':
                try:
                    if not labgroup in r.labgroup:
                        r.labgroup += ', %s' % labgroup
                except TypeError:
                    # if r.labgroup is none, can't check if labgroup is in it
                    r.labgroup = labgroup
            fetch_meta_data(modelspec, r, attrs, cellid)

            if new_entry:
                session.add(r)

            r.cellid = cellid
            session.commit()
            results_id = r.id

    session.close()
    modelspec.cell_index = cellidx

    return results_id


def fetch_meta_data(modelspec, r, attrs, cellid=None):
    """Assign attributes from model fitter object to Results object.

    Arguments:
    ----------
    modelspec : nems modelspec with populated metadata dictionary
        Stack containing meta data, modules, module names et cetera
        (see nems_modules).
    r : sqlalchemy ORM object instance
        Results object, either a blank one that was created before calling
        this function or one that was retrieved via a query to Results.

    Returns:
    --------
    Nothing. Attributes of 'r' are modified in-place.

    """

    r.lastmod = datetime.datetime.now().replace(microsecond=0)

    for a in attrs:
        # list of non-numerical attributes, should be blank instead of 0.0
        if a in ['modelpath', 'modelfile', 'githash']:
            default = ''
        else:
            default = 0.0
        # TODO: hard coded fix for now to match up stack.meta names with
        # results names.
        # Either need to maintain hardcoded list of fields instead of pulling
        # from Results, or keep meta names in fitter matched to columns
        # some other way if naming rules change.
        #if 'fit' in a:
        #    k = a.replace('fit', 'est')
        #elif 'test' in a:
        #    k = a.replace('test', 'val')
        #else:
        #    k = a
        v=_fetch_attr_value(modelspec, a, default, cellid)
        setattr(r, a, v)
        #log.info("modelspec: meta {0}={1}".format(a,v))



def _fetch_attr_value(modelspec, k, default=0.0, cellid=None):
    """Return the value of key 'k' of modelspec.meta, or default."""

    # if modelspec.meta[k] is a string, return it.
    # if it's an ndarray or anything else with indices, get the first index;
    # otherwise, just get the value. Then convert to scalar if np data type.
    # or if key doesn't exist at all, return the default value.
    if k in modelspec.meta:
        v = modelspec.meta[k]
        if not isinstance(v, str):
            try:
                if cellid is not None and type(v==list):
                    cellids = modelspec.meta['cellids']
                    i = [index for index, value in enumerate(cellids) if value == cellid]
                    v = modelspec.meta[k][i[0]]
                else:
                    v = modelspec.meta[k][0]
                    
            except BaseException:
                v = modelspec.meta[k]

            finally:
                try:
                    v = np.asscalar(v)
                    if np.isnan(v):
                        log.warning("value for %s, converting to 0.0 to avoid errors when"
                                    " saving to mysql", k)
                        v = 0.0
                except BaseException:
                    pass
        else:
            v = modelspec.meta[k]
    else:
        v = default
    if isinstance(v, np.ndarray):
        v = v.item()

    return v

def get_batch(name=None, batchid=None):
    # eg, sql="SELECT * from Batches WHERE batch=301"
    engine = Engine()
    params = ()
    sql = "SELECT * FROM sBatch WHERE 1"
    if not batchid is None:
        sql += " AND id=%s"
        params = params+(batchid,)

    if not name is None:
       sql += " AND name like %s"
       params = params+("%"+name+"%",)

    d = pd.read_sql(sql=sql, con=engine, params=params)

    return d

def add_batch_data(cellid, batch, recording_uri_list):
    """
    Save information about a fit based on modelspec.meta
    :param cellid:
    :param batch:
    :param recording_uri_list:
    :return:
    """
    if type(recording_uri_list) is str:
        recording_uri_list = [recording_uri_list]

    db_tables = Tables()
    Batches = db_tables['Batches']
    Data = db_tables['Data']

    session = Session()
    results_id = None

    session = Session()
    db_tables = Tables()
    Batches = db_tables['Batches']
    Data = db_tables['Data']

    r = (session.query(Batches)
         .filter(Batches.cellid == cellid)
         .filter(Batches.batch == batch)
         .first()
         )
    if r is None:
        # add cell/batch to Data
        log.info("Adding (%s/%d) to Batches", cellid, batch)
        r = Batches()
        r.cellid = cellid
        r.batch = batch
        session.add(r)
        session.commit()

    r = (session.query(Data)
         .filter(Data.cellid == cellid)
         .filter(Data.batch == batch)
         )
    if r.count():
        log.info("Deleting existing (%s/%d) entries in Data", cellid, batch)
        for row in r:
            print(row.filepath)
        r.delete()
        session.commit()

    for groupid, uri in enumerate(recording_uri_list):
        r = Data()
        r.cellid = cellid
        r.batch = batch
        r.groupid = groupid + 1
        r.filepath = uri
        r.val_snr = 0
        r.est_snr = 0
        r.min_isolation = 0
        r.min_snr_index = 0
        session.add(r)

    session.commit()

    return True


def get_batch_cells(batch=None, cellid=None, rawid=None, as_list=False):
    '''
    Retrieve a dataframe from Data containing all cellids in a batch.

    Parameters:
    ----------
    batch : int
        The batch number to retrieve cellids from.
    cellid : str
        A full or partial (must include beginning) cellid to match entries to.
        Ex: cellid='AMT' would return all cellids beginning with AMT.
    rawid : int
        A full rawid to match entries to (must be an exact match).
    as_list : bool
        If true, return cellids as a list instead of a dataframe.
        (default False).

    Returns:
    -------
    d : Integer-indexed dataframe with one column for matched cellids and
        one column for batch number.
        If as_list=True, d will instead be a list of cellids.

    '''
    # eg, sql="SELECT * from Batches WHERE batch=301"
    #engine = Engine()
    SQL_ENGINE = get_setting('SQL_ENGINE')
    params = ()
    sql = "SELECT DISTINCT cellid,batch FROM Data WHERE 1"
    if batch is not None:
        sql += " AND batch=%s"
        params = params+(batch,)

    if cellid is not None:
        if SQL_ENGINE == 'sqlite':
            sql += " AND cellid like '%s'"
        else:
            sql += " AND cellid like %s"
        params = params+(cellid+"%",)

    if rawid is not None:
        sql+= " AND rawid = %d"
        params=params+(rawid,)

    #d = pd.read_sql(sql=sql, con=engine, params=params)

    #sql = sql % params
    #print(sql)
    #print(params)

    d = pd_query(sql=sql, params=params)
    if as_list:
        return d['cellid'].values.tolist()
    else:
        return d


def get_batch_cell_data(batch=None, cellid=None, rawid=None, label=None):

    engine = Engine()
    # eg, sql="SELECT * from Data WHERE batch=301 and cellid="
    params = ()
    #" AND substring(Data.cellid,1,locate('_',Data.cellid)-1)=sCellFile.cellid)" +  # remove underscore and beyond from Data.cellid
    sql = ("SELECT DISTINCT Data.*,sCellFile.goodtrials" +
           " FROM Data LEFT JOIN sCellFile " +
           " ON (Data.rawid=sCellFile.rawid " +
           " AND Data.cellid=sCellFile.cellid)" +  # remove underscore and beyond from Data.cellid
           " WHERE 1")
    if batch is not None:
        sql += " AND Data.batch=%s"
        params = params+(batch,)

    if cellid is not None:
        sql += " AND Data.cellid like %s"
        params = params+(cellid+"%",)

    if (rawid is not None):
        sql += " AND Data.rawid IN %s"
        rawid = tuple([str(i) for i in list(rawid)])
        params = params+(rawid,)

    if label is not None:
        sql += " AND Data.label like %s"
        params = params + (label,)
    sql += " ORDER BY Data.filepath"
    #print(sql)
    d = pd.read_sql(sql=sql, con=engine, params=params)
    if label == 'parm':
        d['parm'] = d['filepath']
    else:
        d.set_index(['cellid', 'groupid', 'label', 'rawid', 'goodtrials'], inplace=True)
        d = d['filepath'].unstack('label')

    return d


def get_batches(name=None, verbose=True):
    # eg, sql="SELECT * from Batches WHERE batch=301"
    params = ()
    sql = "SELECT *,id as batch FROM sBatch WHERE 1"
    if name is not None:
        sql += " AND name like %s"
        params = params+("%"+name+"%",)
    d = pd_query(sql=sql, params=params)
    if verbose:
        for i, r in d.iterrows():
            print(f"{r['batch']}: {r['name']} runclassid={r.runclassid}")
    else:
        return d


def get_cell_files(cellid=None, runclass=None, rawid=None):
    # eg, sql="SELECT * from sCellFile WHERE cellid like "TAR010c-30-1"
    engine = Engine()
    params = ()
    sql = ("SELECT sCellFile.*,gRunClass.name, gSingleRaw.isolation FROM sCellFile INNER JOIN "
           "gRunClass on sCellFile.runclassid=gRunClass.id "
           " INNER JOIN "
           "gSingleRaw on sCellFile.rawid=gSingleRaw.rawid and sCellFile.singleid=gSingleRaw.singleid WHERE 1")
    if cellid is not None:
        sql += " AND sCellFile.cellid like %s"
        params = params+("%"+cellid+"%",)
    if runclass is not None:
        sql += " AND gRunClass.name like %s"
        params = params+("%"+runclass+"%",)
    if rawid is not None:
        sql+= " AND sCellFile.rawid = %s"
        params=params+(rawid,)

    d = pd.read_sql(sql=sql, con=engine, params=params)

    return d


# temporary function while we migrate databases
# (don't have access to gRunClass right now, so need to use rawid)
def get_cell_files2(cellid=None, runclass=None, rawid=None):
    engine = Engine()
    params = ()
    sql = ("SELECT sCellFile.* FROM sCellFile WHERE 1")

    if not cellid is None:
        sql += " AND sCellFile.cellid like %s"
        params = params+("%"+cellid+"%",)
    if not runclass is None:
        sql += " AND gRunClass.name like %s"
        params = params+("%"+runclass+"%",)
    if not rawid is None:
        sql+=" AND sCellFile.rawid = %s"
        params = params+(rawid,)


    d = pd.read_sql(sql=sql, con=engine, params=params)

    return d


def get_isolation(cellid=None, batch=None):
    engine = Engine()
    sql = ("SELECT min_isolation FROM Batches WHERE cellid = {0}{1}{2} and batch = {3}".format("'",cellid,"'",batch))

    d = pd.read_sql(sql=sql, con=engine)
    return d


def get_cellids(rawid=None):
    engine = Engine()
    sql = ("SELECT distinct(cellid) FROM sCellFile WHERE 1")

    if rawid is not None:
        sql+=" AND rawid = {0} order by cellid".format(rawid)
    else:
        sys.exit('Must give rawid')

    cellids = pd.read_sql(sql=sql,con=engine)['cellid']

    return cellids


def list_batches(name=None):

    d = get_batches(name)

    for x in range(0, len(d)):
        print("{} {}".format(d['batch'][x], d['name'][x]))

    return d


def get_data_parms(rawid=None, parmfile=None):
    # get parameters stored in gData associated with a rawfile
    engine = Engine()
    if rawid is not None:
        sql = ("SELECT gData.* FROM gData INNER JOIN "
               "gDataRaw ON gData.rawid=gDataRaw.id WHERE gDataRaw.id={0}"
               .format(rawid))
        # sql="SELECT * FROM gData WHERE rawid={0}".format(rawid)

    elif parmfile is not None:
        sql = ("SELECT gData.* FROM gData INNER JOIN gDataRaw ON"
               "gData.rawid=gDataRaw.id WHERE gDataRaw.parmfile = '{0}'"
               .format(parmfile))
        log.info(sql)
    else:
        pass

    d = pd.read_sql(sql=sql, con=engine)

    return d


def get_siteid(s):
    """
    strip off the tail of a cellid to get the siteid
    """
    if '_' in s:
        split_dash = s.split("-")
        split_underscore = s.split("_")
        return split_dash[0] + '*' + split_underscore[-1]
    else:
        return s.split("-")[0]

def get_batch_modelnames(batch, cellid_filter=None, modelname_filter=None, min_count=0):
    
    sql = "SELECT modelname, count(cellid), avg(r_test), avg(r_ceiling), avg(n_parms)" +\
        f" FROM Results WHERE batch={batch}"
    if cellid_filter is not None:
        sql += f" AND cellid like '{cellid_filter}'"
    if modelname_filter is not None:
        sql += f" AND modelname like '%{modelname_filter}%'"
    
    d = pd_query(sql)
    
    return d
    
    
def get_batch_sites(batch, modelname_filter=None):
    """
    get all siteids and a representative cellid from each site in a batch
    :param batch: NEMS batch
    :return: (siteids, cellids) tuple lists of siteids and cellids
    """
    if modelname_filter is None:
        d = get_batch_cells(batch=batch)
    else:
        d = batch_comp(batch=batch, modelnames=[modelname_filter])
        d = d.reset_index()
        
    d['siteid'] = d.cellid.map(get_siteid)

    siteids = list(set(d.siteid.tolist()))

    if batch == 322:
        cellids = [d.loc[d.cellid.str.startswith(s)].cellid.values[0] for s in siteids]
        if 'DRX006b' in siteids:
            siteids.remove('DRX006b')
            siteids.extend(['DRX006b.e1:64', 'DRX006b.e65:128'])
            cellids.append('DRX006b-66-1')
        if 'DRX007a' in siteids:
            siteids.remove('DRX007a')
            siteids.extend(['DRX007a.e1:64', 'DRX007a.e65:128'])
            cellids.append('DRX007a-69-1')
        if 'DRX008b' in siteids:
            siteids.remove('DRX008b')
            siteids.extend(['DRX008b.e1:64', 'DRX008b.e65:128'])
            cellids.append('DRX008b-66-1')
    else:
        if not any(['*' in s for s in siteids]):
            cellids = [d.loc[d.cellid.str.startswith(s)].cellid.values[0] for s in siteids]
        elif all(['*' in s for s in siteids]):
            cellids=[]
            for s in siteids:
                s_split = s.split('*')
                cells_in_site = d.cellid.str.contains(s_split[0]) & d.cellid.str.contains(s_split[1])
                cellids.append(d.loc[cells_in_site].cellid.values[0])
        else:
            raise RuntimeError('siteids should either all have "*" in them, or none of them should. In this case only some do. Why?')

    #sort siteids and cellids by siteids
    sorted_pairs = sorted(zip(siteids, cellids))
    tuples = zip(*sorted_pairs)
    siteids, cellids = [list(tuple) for tuple in tuples]

    return siteids, cellids


def batch_comp(batch=301, modelnames=None, cellids=None, stat='r_test', join_how='left', shortnames=None):
    Results = Tables()['Results']
    if modelnames is None:
        modelnames = ['parm100pt_wcg02_fir15_pupgainctl_fit01_nested5',
                      'parm100pt_wcg02_fir15_pupgain_fit01_nested5',
                      'parm100pt_wcg02_fir15_stategain_fit01_nested5'
                      ]
    if type(batch) is int:
        batches = [batch]
    elif type(batch) is list:
        batches = batch.copy()
    else:
        raise ValueError(f"batch must be integer or list of integers")
    session = Session()
    results=None
    for batch in batches:
        bresults=None
        for mn in modelnames:
            q = (session.query(Results)
                 .filter(Results.batch == batch)
                 .filter(Results.modelname == mn))
            if cellids is not None:
                q = q.filter(Results.cellid.in_(cellids))
            tr = psql.read_sql_query(q.statement, session.bind)
            tc = tr[['cellid',stat]]
            tc = tc.set_index('cellid')
            tc.columns = [mn]
            if bresults is None:
                bresults = tc
            else:
                bresults=bresults.join(tc, how=join_how)
        if results is None:
            results=bresults
        else:
            results = pd.concat((results,bresults), axis=0)
    session.close()
    if shortnames is not None:
        results.columns=shortnames
    return results


def get_results_file(batch, modelnames=None, cellids=None):
    Results = Tables()['Results']
    session = Session()
    query = (
        session.query(Results)
        .filter(Results.batch == batch)
        .order_by(desc(Results.lastmod))
        )

    if modelnames is not None:
        if not isinstance(modelnames, list):
            raise ValueError("Modelnames should be specified as a list, "
                             "got %s", str(type(modelnames)))
        query = query.filter(Results.modelname.in_(modelnames))

    if cellids is not None:
        if not isinstance(cellids, list):
            raise ValueError("Cellids should be specified as a list, "
                             "got %s", str(type(cellids)))
        query = query.filter(Results.cellid.in_(cellids))

    results = psql.read_sql_query(query.statement, session.bind)
    session.close()

    if results.empty:
        raise ValueError("No result exists for:\n"
                         "batch: {0}\nmodelnames: {1}\ncellids: {2}\n"
                         .format(batch, modelnames, cellids))
    else:
        return results

def export_fits(batch, modelnames=None, cellids=None, dest=None):
    """

    :param batch: required
    :param modelnames: required - [list of modelnames]
    :param cellids: [list of cellids] - default all cells
    :param dest: path where exported models should be saved
    :return:
    """
    if modelnames is None:
        raise ValueError('currently must specify modelname list')
    if dest is None:
        raise ValueError('currently must specify dest path')
    if not os.path.exists(dest):
        raise ValueError('dest path {} does not exist'.format(dest))

    if type(modelnames) is str:
        modelnames=[modelnames]

    df = get_results_file(batch, modelnames=modelnames, cellids=cellids)
    for index, row in df.iterrows():
        modelpath = row['modelpath']
        subdir = row['cellid'] + '_' + os.path.basename(modelpath)
        outpath = os.path.join(dest, subdir)
        log.info('copy {} to {}'.format(modelpath, outpath))
        if not os.path.exists(outpath):
            shutil.copytree(modelpath, outpath)
        else:
            log.info('{} already exists, skipping'.format(subdir))

    return dest


def get_stable_batch_cells(batch=None, cellid=None, rawid=None, label ='parm'):
    '''
    New functionality as of 03/13/2020
        Previous: returned set of cellids that was stable across all files
            e.g. if 20 cells were recorded across four files but only 15 of the cells
            were isolated across all four files, this would return a list of only
            those 15 cells and the rawid for all four files.

        New behavior: Now, does the opposite of before and returns the 20 cellids
            and only the three file rawids where all 20 cells were stable.

    If rawid is passed, this will only return the cells that were stable across those
    rawids. So, in the example, if you passed all four rawids you'd only get the 15
    cellids back corresponding to the stable isolation units.
    '''
    if (batch is None) | (cellid is None):
        raise ValueError

    engine = Engine()
    params = ()
    sql = "SELECT cellid, rawid FROM Data WHERE 1"

    if batch is not None:
        sql += " AND batch=%s"
        params += (batch,)

    if label is not None:
       sql += " AND label = %s"
       params += (label,)

    if cellid is not None:
        if type(cellid) is str:
            sql += " AND cellid like %s"
            params += ("%"+cellid+"%",)
        else:
            cellid = tuple([c for c in cellid])
            sql += " AND cellid IN %s"
            params += (cellid,)

    # check to see if rawid was specified, if not, just find the
    # superset of cells
    if rawid is not None:
        try:
            rawid = tuple([rid for rid in rawid])
        except TypeError:
            rawid = (rawid,)
        sql += " AND rawid IN %s"
        params += (rawid,)
        d = pd.read_sql(sql=sql, con=engine, params=params)
        cellids = d.cellid.unique().tolist()
        rawid = d.rawid.unique().tolist()

        # keep only cellids present for ALL rawids
        cellids = [cid for cid in cellids if np.all(np.sort(d[d.cellid==cid].rawid.unique().tolist())==np.sort(rawid))]

        return cellids, rawid

    else:
        d = pd.read_sql(sql=sql, con=engine, params=params)
        cellids = d.cellid.unique().tolist()
        rawid = d.rawid.unique().tolist()
        # get only rawid that exist for ALL the cells if no rawid specification was passed to the fn
        rawid = [rid for rid in rawid if d[d.rawid==rid].cellid.unique().tolist()==cellids]

        if (len(d.rawid.unique()) == 2) & (len(rawid) == 0):
            # keep rawid with the most cells if not equal
            rawid = [int(d.groupby(by='rawid').count().idxmax().values[0])]
            cellids = d[d.rawid.isin(rawid)].cellid.unique().tolist()

        return cellids, rawid


def get_wft(cellid=None):
    engine = Engine()
    params = ()
    sql = "SELECT meta_data FROM gSingleCell WHERE 1"

    sql += " and cellid =%s"
    params = params+(cellid,)

    d = pd.read_sql(sql=sql, con=engine, params=params)
    if d.values[0][0] is None:
        log.info('no meta_data information for {0}'.format(cellid))
        return -1

    wft = json.loads(d.values[0][0])
    ## 1 is fast spiking, 0 is regular spiking
    celltype = int(wft['wft_celltype'])

    return celltype


def get_gSingleCell_meta(cellid=None, fields=None):

    engine = Engine()
    params = ()
    sql = "SELECT meta_data FROM gSingleCell WHERE 1"

    sql += " and cellid =%s"
    params = params+(cellid,)

    d = pd.read_sql(sql=sql, con=engine, params=params)
    if d.values[0][0] is None:
        print('no meta_data information for {0}'.format(cellid))
        return -1
    else:
        dic = json.loads(d.values[0][0])
        if type(fields) is list:
            out = {}
            for f in fields:
                out[f] = dic[f]

        elif type(fields) is str:
            out = dic[fields]
        elif fields is None:
            out = {}
            fields = dic.keys()
            for f in fields:
                out[f] = dic[f]

        return out

def get_rawid(cellid, run_num):
    """
    Used to return the rawid corresponding to given run number. To be used if
    you have two files at a given site that belong to the same batch but were
    sorted separately and you only want to load cellids from one of the files.

    ex. usage in practice would be to pass a sys arg cellid followed by the
    run_num:

        cellid = 'TAR017b-04-1_04'

        This specifies cellid and run_num. So parse this string and pass as args
        to this function to return rawid
    """
    engine = Engine()
    params = ()
    sql = "SELECT rawid FROM sCellFile WHERE 1"

    if cellid is not None:
        sql += " AND cellid like %s"
        params = params+(cellid+"%",)

    if run_num is not None:
        sql += " AND respfile like %s"
        params = params+(cellid[:-5]+run_num+"%",)

    d = pd.read_sql(sql=sql, con=engine, params=params)

    return [d['rawid'].values[0]]

def get_pen_location(cellid):
    """
    Cellid can be string or list. For every channel in the list, return the
    well position. For example, if cellid = ['AMT024a-01-2', 'AMT024a-03-2']
    then this code expects there to be at least 3 well positions in the db,
    it will return the 0th and 2nd positions.

    DO NOT pass list of cellids from different sites. This will not work

    If recording with a single probe array, there is only one well position for
    all 64 channels. For this reason, it doesn't make sense for cellid to be a
    list
    """
    engine = Engine()
    params = ()
    sql = "SELECT wellposition FROM gPenetration WHERE penname like '{}'"

    if type(cellid) is list:
        penname = cellid[0][:6]
    if type(cellid is np.ndarray):
        penname = cellid[0][:6]
    if type(cellid) is str:
        penname = cellid[:6]

    sql = sql.format(penname)

    d = pd.read_sql(sql=sql, con=engine)
    xy = d.values[0][0].split('+')
    # return table of x y values
    if type(cellid) is str:
        table_xy = pd.DataFrame(index=[cellid], columns=['x', 'y'])
    else:
        # only keep unique chans
        cellid = np.unique([c[:10] for c in cellid])
        table_xy = pd.DataFrame(index=cellid, columns=['x', 'y'])

    for i, pos in enumerate(xy):
        vals = pos.split(',')
        if (len(vals) > 1) & (i < len(cellid)):
            table_xy['x'][i] = int(vals[0])
            table_xy['y'][i] = int(vals[1])

    return table_xy


#### Data management

def save_recording_to_db(recfilepath, meta=None, user="nems", labgroup="",
                         public=True):
    """
    expects recfilepath == "/path/to/data/<exptname>_<hash>.tgz"

    """
    engine = Engine()
    conn = engine.connect()

    path, base = os.path.split(recfilepath)
    base = base.split("_")
    pre = base[0]
    hsh = base[1].split(".")[0]
    batch = int(meta.get("batch", 0))
    if batch > 0:
        path, batchstr = os.path.split(path)

    file_hash = utils.recording_filename_hash(name=pre, meta=meta, uri_path=path,
                                        uncompressed=False)
    meta_string = json.dumps(meta, sort_keys=True)

    if file_hash != recfilepath:
        raise ValueError("meta does not produce hash matching recfilepath")

    sql = "INSERT INTO Data (batch,hash,meta,filepath,label," + \
          "username,labgroup,public) VALUES" + \
          " ({},'{}','{}','{}','{}','{}','{}',{})"
    sql = sql.format(batch, hsh, meta_string, recfilepath, "recording",
                     user, labgroup, int(public))
    r = conn.execute(sql)
    dataid = r.lastrowid
    log.info("Added new entry %d for: %s.", dataid, recfilepath)

    return dataid

def recent_models(batch,N=20):
    # Loads all results for a given batch, prints the mean r_test for the last N (20).
    assert type(batch) is int
    df = pd_query(f"SELECT * from Results WHERE batch={batch}").sort_values(by='lastmod')
    # df=df[df['cellid'].isin(cellids)]
    dfm = df.groupby(by='modelname').last().sort_values(by='lastmod')
    df_mean = df[['modelname', 'r_test', 'r_fit']].groupby(by='modelname').agg('mean')
    df_count = df[['modelname']].groupby(by='modelname').size()
    dfm['r_test'] = df_mean['r_test'].round(3)
    dfm['r_fit'] = df_mean['r_fit'].round(3)
    dfm['count'] = df[['modelname']].groupby(by='modelname').size()
    print(dfm[['r_test', 'r_fit', 'count']].tail(N).to_string())
    return dfm
