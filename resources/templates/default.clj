;;(doctype :xhtml-transitional)
[:html
 {:xmlns "http://www.w3.org/1999/xhtml" :lang "en" :xml:lang "en"}
 [:head
  [:meta {:http-equiv "content-type" :content "text/html; charset=UTF-8"}]
  [:meta {:name "description" :content (:description metadata)}]
  [:meta {:name "keywords" :content (:tags metadata)}]
  [:meta {:name "author" :content "Nurullah Akkaya"}]
  [:link {:rel "icon" :href "/images/favicon.ico" :type "image/x-icon"}]
  [:link {:rel "shortcut icon" :href "/images/favicon.ico" :type "image/x-icon"}]
  [:link {:rel "stylesheet" :type "text/css" :href "/css/bootstrap.css"}]
  [:link {:rel "stylesheet" :type "text/css" :href "/css/font-awesome.min.css"}]
  [:link {:rel "stylesheet" :type "text/css" :href "/css/crsmithdev.css"}]
  [:title (:title metadata)]]
 [:body
  [:div {:class "header"}
   [:nav {:class "navbar navbar-fixed-top" :role "navigation"}
    [:div {:class "container"}
     [:div {:class "navbar-header"}
      [:button {:type "button" :class "navbar-toggle" :data-toggle "collapse"
                :data-target ".navbar-ex1-collapse"}
      [:span {:class "sr-only"} "Toggle navigation"]
      [:span {:class "icon-bar"}]
      [:span {:class "icon-bar"}]
      [:span {:class "icon-bar"}]]
      [:a {:class "navbar-brand" :href "/index.html"} "Chris Smith"]]
     [:div {:class "collapse navbar-collapse navbar-ex1-collapse"}
      [:ul {:class "nav navbar-nav"}
       [:li [:a {:href "/blog.html"} "Blog"]]
       [:li [:a {:href "/projects.html"} "Projects"]]
       [:li [:a {:href "/about.html"} "About"]]]
      [:ul {:class "nav navbar-nav pull-right"}
       [:li [:a {:href "http://github.com/crsmithdev" :class "icon-header"} [:i {:class "icon-github icon-2x"}]]
       [:li [:a {:href "http://twitter.com/crsmithdev" :class "icon-header"} [:i {:class "icon-twitter icon-2x"}]]]]]]]]]
  [:div {:class "content"}
   [:div {:class "container"}
    content]]
  [:div {:class "footer"}
   [:div {:class "container"}
    [:p "Built with "
     [:a {:href "http://getbootstrap.com/"} "Bootstrap"]
     " and " 
     [:a {:ref "https://github.com/nakkaya/static"} "Static"]
     [:br]
     "&copy; 2013 "
     [:a {:href "http://crsmithdev.com"} "Chris Smith"]]]]]]
