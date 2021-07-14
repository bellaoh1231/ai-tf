# from typing import Any
from flask import Flask, request, session,g,redirect,url_for, abort, flash, render_template, jsonify, json
# import pandas as pd
import matplotlib.pyplot as plt
import sqlite3

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index_yh.html")


@app.route('/index4.html')
def index4():

    conn = sqlite3.connect('trest.db')
    curs = conn.cursor()

    sql = "select no2 from address"
    curs.execute(sql)
    rows = curs.fetchall() #no2 db 저장을 rows에 저장 됨

    return render_template("index4.html", add2=rows, add22=rows)


@app.route("/test", methods=["POST"])
def languages():
    value1 = request.form["itemid1"]
    value2 = request.form["itemid2"]
    value3 = request.form["itemid3"]
    value4 = request.form["itemid4"]
    value5 = request.form["itemid5"]
    value6 = request.form["itemid6"]

    print(value1)
    print(value4)
    print("데이터 넘어왔나?")

    conn = sqlite3.connect("trest.db")
    curs = conn.cursor()
    curs.execute('SELECT * FROM sampledb where 1=1 and (?="" OR 세대수=?) and (?="선택하세요" OR 지역=?) and (?="선택하세요" OR 건물유형=?)',
                 (value6, value6, value4, value4, value5, value5))  # test1에 있는 값을 조건 없이 불러옴 or조건으로 입력값이 있는 것만 조회함

    # row_headers = [x[0] for x in curs.description]
    rows = curs.fetchall()
    # print(rows)
    print("test2")

    curs.execute('SELECT * from piechart')
    chart_value = curs.fetchall()
    chart_value[0] = list(chart_value[0])
    print(chart_value[0])
    # print(type(chart_value[0][0]))

    return jsonify(rows, chart_value[0])


@app.route("/test2", methods=["POST"])
def languages2():
    value1 = request.form["itemid1"]
    value2 = request.form["itemid2"]
    value3 = request.form["itemid3"]
    value4 = request.form["itemid4"]
    value5 = request.form["itemid5"]
    value6 = request.form["itemid6"]

    print(value1)
    print(value4)
    print("데이터 넘어왔나?")

    conn = sqlite3.connect("trest.db")
    curs = conn.cursor()
    curs.execute('SELECT 연면적 FROM sampledb where 1=1 and (?="" OR 세대수=?) and (?="선택하세요" OR 지역=?) and (?="선택하세요" OR 건물유형=?)',
                 (value6, value6, value4, value4, value5, value5))  # test1에 있는 값을 조건 없이 불러옴 or조건으로 입력값이 있는 것만 조회함

    # row_headers = [x[0] for x in curs.description]
    rows = curs.fetchall()
    # print(rows)
    print("test2")
    return jsonify(rows)


@app.route('/add', methods=['POST'])
def add():
    value1 = request.form['x_py']
    print(value1)
    conn = sqlite3.connect('trest.db')
    curs = conn.cursor()

    sql = "select no3 from address where no2='" + value1 + "'"
    print(sql)
    curs.execute(sql)
    rows = curs.fetchall()
    print(rows)
    return jsonify(rows, value1)


if __name__ == "__main__":
    app.run()