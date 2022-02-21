#
# The main search hooks for the Search Flask application.
#
from flask import (
    Blueprint, redirect, render_template, request, url_for
)

from week1.opensearch import get_opensearch

bp = Blueprint('search', __name__, url_prefix='/search')


# Process the filters requested by the user and return a tuple that is appropriate for use in: the query, URLs displaying the filter and the display of the applied filters
# filters -- convert the URL GET structure into an OpenSearch filter query
# display_filters -- return an array of filters that are applied that is appropriate for display
# applied_filters -- return a String that is appropriate for inclusion in a URL as part of a query string.  This is basically the same as the input query string
def process_filters(filters_input):
    # Filters look like: &filter.name=regularPrice&regularPrice.key={{ agg.key }}&regularPrice.from={{ agg.from }}&regularPrice.to={{ agg.to }}
    filters = []
    display_filters = []  # Also create the text we will use to display the filters that are applied
    applied_filters = ""
    for filter in filters_input:
        type = request.args.get(filter + ".type")
        display_name = request.args.get(filter + ".displayName", filter)
        #
        # We need to capture and return what filters are already applied so they can be automatically added to any existing links we display in aggregations.jinja2
        applied_filters += "&filter.name={}&{}.type={}&{}.displayName={}".format(filter, filter, type, filter,
                                                                                 display_name)
        #TODO: IMPLEMENT AND SET filters, display_filters and applied_filters.
        # filters get used in create_query below.  display_filters gets used by display_filters.jinja2 and applied_filters gets used by aggregations.jinja2 (and any other links that would execute a search.)
        if type == "range":
            from_val = request.args.get(filter + ".from")
            to_val = request.args.get(filter + ".to")
            range = {}
            if from_val:
                range["gte"] = from_val
            if to_val:
                range["lt"] = to_val
            filters.append({"range": {filter: range}})
            display_filters.append(f"{display_name}: {from_val} to {to_val}")
            applied_filters += f"&{filter}.from={from_val}&{filter}.to={to_val}"

        elif type == "terms": 
            field_name = request.args.get(filter + ".name")
            field_val  = request.args.get(filter + ".key")
            filters.append({"term": {filter: field_val}})
            display_filters.append(f"{field_name}: {field_val}")
            applied_filters += f"&{filter}.name={field_name}&{filter}.key={field_val}"

    print("Filters: {}".format(filters))

    return filters, display_filters, applied_filters


# Our main query route.  Accepts POST (via the Search box) and GETs via the clicks on aggregations/facets
@bp.route('/query', methods=['GET', 'POST'])
def query():
    opensearch = get_opensearch() # Load up our OpenSearch client from the opensearch.py file.
    # Put in your code to query opensearch.  Set error as appropriate.
    error = None
    user_query = None
    query_obj = None
    display_filters = None
    applied_filters = ""
    filters = None
    sort = "_score"
    sortDir = "desc"
    if request.method == 'POST':  # a query has been submitted
        user_query = request.form['query']
        if not user_query:
            user_query = "*"
        sort = request.form["sort"]
        if not sort:
            sort = "_score"
        sortDir = request.form["sortDir"]
        if not sortDir:
            sortDir = "desc"
        query_obj = create_query(user_query, [], sort, sortDir)
    elif request.method == 'GET':  # Handle the case where there is no query or just loading the page
        user_query = request.args.get("query", "*")
        filters_input = request.args.getlist("filter.name")
        sort = request.args.get("sort", sort)
        sortDir = request.args.get("sortDir", sortDir)
        if filters_input:
            (filters, display_filters, applied_filters) = process_filters(filters_input)

        query_obj = create_query(user_query, filters, sort, sortDir)
    else:
        query_obj = create_query("*", [], sort, sortDir)

    print("query obj: {}".format(query_obj))

    index_name = "bbuy_products"
    response = opensearch.search(
            body=query_obj,
            index=index_name
        )

    #print(response)
    if error is None:
        return render_template("search_results.jinja2", query=user_query, search_response=response,
                               display_filters=display_filters, applied_filters=applied_filters,
                               sort=sort, sortDir=sortDir)
    else:
        redirect(url_for("index"))


def create_query(user_query, filters, sort="_score", sortDir="desc"):
    print("Query: {} Filters: {} Sort: {}".format(user_query, filters, sort))
    
    query_obj = {
                    "size": 10,
                    "sort":
                    [
                        {sort: {"order": sortDir}}
                    ],
                    "query":
                    {
                        "function_score":
                        {
                            "query": 
                            {
                                "bool": 
                                {
                                    "must": 
                                    [
                                        
                                        {
                                            "multi_match": 
                                            { 
                                                "query": user_query, 
                                                "fields": ['name^10', 'shorDescription^4', 'longDescription^2','department']  
                                            } 
                                        }
                                    ],
                                    "should":
                                    [
                                        {
                                            "term": 
                                            {
                                                "name": 
                                                {
                                                    "value": user_query,
                                                    "boost": 10.0
                                                }
                                            }
                                        }

                                    ],
                                    "filter": filters,
                                }
                            },
                            "boost_mode": "multiply",
                            "score_mode": "avg",
                            "functions": 
                            [
                                {
                                    "field_value_factor": {
                                        "field": "salesRankShortTerm",
                                        "modifier": "reciprocal",
                                        "missing": 100000000
                                    }
                                },
                                {
                                    "field_value_factor": {
                                        "field": "salesRankMediumTerm",
                                        "modifier": "reciprocal",
                                        "missing": 100000000
                                    }
                                },
                                {
                                    "field_value_factor": {
                                        "field": "salesRankLongTerm",
                                        "modifier": "reciprocal",
                                        "missing": 100000000
                                    }
                                }
                            ]
                        }
                    },
                
                    "aggs": 
                    {
                        "regularPrice": 
                        {
                            "range": 
                            {
                                "field": "salePrice",
                                "ranges": 
                                [
                                    {"to": 100},
                                    {"from": 100, "to": 500},
                                    {"from": 500},
                                ]
                            },
                            "aggs": 
                            {
                                "price_stats": 
                                {
                                    "stats": {"field": "regularPrice"}
                                }
                            }
                        },
                        "department":
                        {
                            "terms":
                            {
                                "field": "department.keyword",
                                "min_doc_count":0
                            } 

                        },
                        "missing_images": 
                        {
                            "missing": 
                            {
                                "field": "image"
                            }
                        }
                    }                    

                            
                }
    
    if user_query == "*":
        query_obj = {
            "query":
                    {
                        "function_score":
                        {
                            "query": 
                            {
                                "bool": 
                                {
                                    "must": 
                                    [
                                        
                                        {
                                            "match_all": {}
                                        }
                                    ],
                                    "filter": filters,
                                }
                            }
                        }
                    },
                    "aggs": 
                    {
                        "regularPrice": 
                        {
                            "range": 
                            {
                                "field": "salePrice",
                                "ranges": 
                                [
                                    {"to": 100},
                                    {"from": 100, "to": 500},
                                    {"from": 500},
                                ]
                            },
                            "aggs": 
                            {
                                "price_stats": 
                                {
                                    "stats": {"field": "regularPrice"}
                                }
                            }
                        },
                        "department":
                        {
                            "terms":
                            {
                                "field": "department.keyword",
                                "min_doc_count":0
                            } 

                        },
                        "missing_images": 
                        {
                            "missing": 
                            {
                                "field": "image"
                            }
                        }
                    }
        }
    return query_obj
