{:title "crsmithdev.com"}

[:div
 [:div {:class "row"}
  [:div {:class "col-md-12"}
   [:h1 "[needs a title]"]]]
 [:div {:class "row"}
  [:div {:class "col-md-6"}
   [:p {:class "lead text-muted"}
    "Writings on code, startups and living in San Francisco."]]]
 [:div {:class "row"}
  [:div {:class "col-md-12"}
   [:p
    "Hi, I'm Chris, an engineer from the midwest now living and working in the heart of downtown San Francisco. "
    "I spend my days hacking for " [:a {:href "http://cir.ca"} "Circa"] ", developing and scaling APIs and other backend systems. "
    "In my own time, I work on many side " [:a {:href "/projects.html"} "projects"] ", study architecture, functional programming, "
    "languages, concurrency and scalability. My current tools of choice are Python, Scala, and Clojure.  This is both a personal "
    "portfolio as well as a " [:a {:href "/blog.html"} "blog"] " of some of the interesting and enlightening things I've discovered."]]]
  [:p
   "I'm on " [:a {:href "http://twitter.com/crsmithdev"} "twitter"] " and " [:a {:href "http://github.com/crsmithdev"} "github"]
   ", and you can also contact me " [:a {:href "mailto:crsmithdev@gmail.com"} "directly"] "."]
 [:div {:class "row"}
  [:div {:class "col-md-6"}
   [:h4 "Recent Blog Posts"]
   (map #(let [f % url (static.core/post-url f)
               [metadata _] (static.io/read-doc f)
               date (static.core/parse-date
                     "yyyy-MM-dd" "dd MMMM yyyy"
                     (re-find #"\d*-\d*-\d*" (str f)))]
      [:div
       [:div [:a {:href url} (:title metadata)]
       [:div date]]])
      (take 8 (reverse (static.io/list-files :posts))))]
    [:div {:class "col-md-6"}
    [:h4 "Recent Activity on Github"]
    [:div {:class "gh-recent"}]]]]
